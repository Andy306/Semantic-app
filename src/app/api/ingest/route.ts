import { handleBootstrapping } from "@/app/services/bootstrap";
import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  const { targetIndex } = await request.json();
  await handleBootstrapping(targetIndex);
  return NextResponse.json({ message: "Ingestion initiated" });
}