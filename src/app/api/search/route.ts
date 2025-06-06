import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { VoyageEmbeddings } from "@langchain/community/embeddings/voyage";
import {PineconeStore} from "@langchain/pinecone";

export async function POST(req: Request) { 
  const { query } = await req.json();

  if(!query) {
    return NextResponse.json({ error: "Query is required" }, { status: 400 });
  }

  try {
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    const voyageEmbeddings = new VoyageEmbeddings({
      apiKey: process.env.VOYAGE_API_KEY!,
      inputType: "query",
      modelName: "voyage-law-2",
    });

    const vectorStore = new PineconeStore(voyageEmbeddings, {
      pineconeIndex: pc.Index(process.env.PINECONE_INDEX as string),
      namespace: process.env.PINECONE_NAMESPACE as string,
    });

    console.log(`query is: ${query}`);

    const retrieved = await vectorStore.maxMarginalRelevanceSearch(query, {
      k: 20,
    }
    );

    const results: any = retrieved.filter((result, index) => {
      return (
        index ===
        retrieved.findIndex((otherResult: any) => {
          return result.metadata.id === otherResult.metadata.id;
        })
      )
    });

    return NextResponse.json({ results }, { status: 200 });

  } catch (error) {
    console.error("Error performing similarity search:", error);
    return NextResponse.json(
      { error: "Error performing similarity search" },
      { status: 500 }
    );

  }
}