"use server";
import { createIndexIfNecessary,pinconeIndexHasVectors } from "@/app/services/pinecone";
import { NextResponse } from "next/server";
import path from "path";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { promises as fs } from "fs";
import { v4 as uuidv4 } from "uuid";
import {VoyageEmbeddings} from "@langchain/community/embeddings/voyage";
import { Document } from "langchain/document";
import { Pinecone } from "@pinecone-database/pinecone";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";


const readMetadata = async (): Promise<Document["metadata"][]> => {
 try {
   const filePath = path.resolve(process.cwd(), "docs/db.json");
   const data = await fs.readFile(filePath, "utf-8");
   const parsed = JSON.parse(data);
   return parsed.documents || [];
   

 } catch (error) {
   console.log("Error reading metadata:", error);
    return [];
 } 
}

const flattenMetadata = (metadata: any): Document["metadata"] => { 
  const flatMetadata = { ...metadata };
  if(flatMetadata.pdf) {
    if (flatMetadata.pdf.pageCount) {
      flatMetadata.totalPages = flatMetadata.pdf.pageCount;
    
    }
    delete flatMetadata.pdf;
  }
  if (flatMetadata.loc) {
    delete flatMetadata.loc;
  }
  return flatMetadata;
}

const batchUpsert = async (
  index: any,
  vectors: any[],
  batchSize: number = 50
) => {
  for (let i = 0; i < vectors.length; i += batchSize) {
    const batch = vectors.slice(i, i + batchSize);
    console.log(`Upserting batch ${i + 1} of ${batch.length} vectors...`);
    await index.upsert(batch);
  }
};


export const initiateBootstrapping = async (targetIndex: string) => {
  const baseURL = process.env.PRODUCTION_URL ? `https://${process.env.PRODUCTION_URL}` : "http://localhost:3000";
  const response = await fetch(`${baseURL}/api/ingest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ targetIndex }),
  });
  if (!response.ok) {
    throw new Error("Failed to initiate bootstrapping");
  }

};

const isValidContent = (content: string): boolean => {
  if (!content || typeof content !== "string") return false;
  const trimmedContent = content.trim();
  return trimmedContent.length > 0 && trimmedContent.length <= 8192;
};

export const handleBootstrapping = async (targetIndex: string) => {
  try {
    console.log(`Running bootstrapping procedure against Pinecone index: ${targetIndex}`);

    await createIndexIfNecessary(targetIndex);
    const hasVectors = await pinconeIndexHasVectors(targetIndex);

    if (hasVectors) {
      console.log(`Pinecone index ${targetIndex} already has vectors. Skipping bootstrapping.`);
      return NextResponse.json({ message: "Pinecone index already has vectors. Skipping bootstrapping." });
    }

    console.log("loading documents and metadata....")

    const docsPath = path.resolve(process.cwd(), "docs/");
    const loader = new DirectoryLoader(docsPath, {
      ".pdf": (filePath: string) => new PDFLoader(filePath),
    });

    const documents = await loader.load();
    if (documents.length === 0) {
      console.log("No documents found in the specified directory.");
      return NextResponse.json({ message: "No documents found in the specified directory." });
    }
    const metadata = await readMetadata();
    const validDocuments = documents.filter((doc) =>
      isValidContent(doc.pageContent)
    );

    validDocuments.forEach((doc) => {
      const fileMetadata = metadata.find(
        (meta) => meta.filename === path.basename(doc.metadata.source));
      if (fileMetadata) {
        doc.metadata = {
          ...doc.metadata,
          ...fileMetadata,
          pageContent: doc.pageContent,
        };
      }
    });

    console.log(`Loaded ${validDocuments.length} valid documents.`);

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const splits = await splitter.splitDocuments(validDocuments);
    console.log(`Split documents into ${splits.length} chunks.`);

    const BATCH_SIZE = 5;

    for (let i = 0; i < splits.length; i += BATCH_SIZE) {
      const batch = splits.slice(i, i + BATCH_SIZE);
      console.log(`Ingesting batch ${Math.floor(i / BATCH_SIZE) + 1} of ${Math.ceil(splits.length / BATCH_SIZE)}`);

    

      const validBatch = batch.filter((split) =>
        isValidContent(split.pageContent)
      );
      if (validBatch.length === 0) {
        console.log("Skipping batch - no valid content found.");
        continue;
      }

      const castedBatch: Document[] = validBatch.map((split) => ({
        pageContent: split.pageContent.trim(),
        metadata: {
          ...flattenMetadata(split.metadata as Document["metadata"]),
          id: uuidv4(),
          pageContent: split.pageContent.trim(),
        },
      }));

      try {
        //generate embeddings and upsert to pinecone
        const voyageEmbeddings = new VoyageEmbeddings({
          apiKey: process.env.VOYAGE_API_KEY,
          inputType: "document",
          modelName: "voyage-law-2",
        });

        const pageContents = castedBatch.map((split) => split.pageContent);
        console.log(`Generating embeddings for ${pageContents.length} chunks...`);

        const embeddings = await voyageEmbeddings.embedDocuments(pageContents);

        if (!embeddings || embeddings.length !== pageContents.length) {
          console.error("Error generating embeddings: Mismatch in number of embeddings generated.", {
            expected: pageContents.length,
            received: embeddings?.length,
          });
          continue;
        }

        //create vectors
        const vectors = castedBatch.map((split, index) => ({
          id: split.metadata.id!,
          values: embeddings[index],
          metadata: split.metadata, 
        }));

        const pc = new Pinecone({
          apiKey: process.env.PINCONE_API_KEY!,
        });

        const index = pc.Index(targetIndex);
        await batchUpsert(index, vectors, 2);

        await new Promise((resolve) => setTimeout(resolve, 1000));
        
      } catch (error) {
        console.error(
          `Error processing batch ${Math.floor(i / BATCH_SIZE) + 1}:`,
          {
            error: error instanceof Error ? error.message : error,
            batchSize: castedBatch.length,
          }
        );
        continue;
      }

    }

    } catch (error: any) {
    console.error("Error occurred during bootstrapping:", {
      message: error.message,
      cause: error.cause?.message,
      stack: error.stack,
    });
    if(error.code === "UND_ERR_CONNECT_TIMEOUT") {
      return NextResponse.json(
        { error: "Connection timeout. Please try again later." },
        { status: 500 }
      );
    }

  }
}



