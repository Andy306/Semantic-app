import { Pinecone } from "@pinecone-database/pinecone";

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY as string
});

export async function createIndexIfNecessary(indexName: string) {
  await pinecone.createIndex(
    {
      name: indexName,
      dimension: 1024,
      spec: {
        serverless: {
          cloud: "aws",
          region: "us-east-1",
        },
      },
      waitUntilReady: true,
      suppressConflicts: true,
    }
  );
}

export async function pinconeIndexHasVectors(indexName: string): Promise<boolean> {
  try {
    const targetIndex = pinecone.index(indexName)

    const stats = await targetIndex.describeIndexStats();

    return (stats.totalRecordCount && stats.totalRecordCount > 0) ? true : false;
  } catch (error) {
    console.error("Error checking index stats:", error);
    return false;
  }

}