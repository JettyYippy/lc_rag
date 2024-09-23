// Load environment variables using ES module syntax
import 'dotenv/config';

// Import necessary modules
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";

// Initialize Pinecone client
const client = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});

const pineconeIndex = client.Index("gen-qa-openai-fast"); // Replace with your Pinecone index name

// Use OpenAI embeddings for query embeddings
const embeddings = new OpenAIEmbeddings();

// Create a retriever from Pinecone
async function pineconeRetriever(query, topK = 5) {
  // Embed the query using OpenAIEmbeddings
  const embeddedQuery = await embeddings.embedQuery(query);

  // Perform a vector search in Pinecone
  const queryResponse = await pineconeIndex.query({
    topK,
    vector: embeddedQuery,
    includeMetadata: true,
  });

  // Extract the relevant documents from the query response
  return queryResponse.matches.map((match) => ({
    text: match.metadata.text,
    score: match.score,
  }));
}

// Define a prompt template with both 'context' and 'question' variables
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant. Use the following information to answer the question."],
  ["human", "Context:\n{context}\n\nQuestion: {question}"],
]);

// Set up the RAG chain
const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", temperature: 0 });
const ragChain = await createStuffDocumentsChain({
  llm,
  prompt: promptTemplate,  // Use the updated prompt template with 'context'
  outputParser: new StringOutputParser(),
});

// Function to retrieve and generate using RAG
async function retrieveAndGenerate(query) {
  // Retrieve relevant documents from Pinecone
  const contextDocs = await pineconeRetriever(query);

  // Check the structure of the documents retrieved
  //console.log("Context Docs:", contextDocs);

  // Prepare the documents for the ragChain
  const formattedDocs = contextDocs.map((doc) => ({ text: doc.text }));

  // Ensure the documents array is passed correctly to the ragChain
  const response = await ragChain.invoke({
    question: query,
    input_documents: formattedDocs  // Change `documents` to `input_documents`
  });

  console.log("Formatted Docs:", formattedDocs);

  return response;
}

// Example usage
(async () => {
  const question = "What is Task Decomposition?";
  const answer = await retrieveAndGenerate(question);
  console.log(answer);
})();
