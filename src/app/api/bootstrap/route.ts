import { initiateBootstrapping } from "@/app/services/bootstrap";
import { NextResponse } from "next/server";

export async function POST() {
  await initiateBootstrapping(process.env.PINECONE_INDEX as string);
  return NextResponse.json({ message: "Bootstrapping initiated" });
}