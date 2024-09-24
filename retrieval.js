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

async function retrieveAndGenerate(query) {
  const contextDocs = await pineconeRetriever(query);
  
  //console.log("Retrieved Context Docs:", contextDocs);
  
  if (!Array.isArray(contextDocs) || contextDocs.length === 0) {
    throw new Error("No documents retrieved from Pinecone.");
  }

  const formattedDocs = contextDocs.map((doc) => ({ text: doc.text }));
  
  //console.log("Formatted Docs:", formattedDocs);

  // Log the invocation parameters
  console.log("Invoking ragChain with:", {
    question: query,
    context: formattedDocs,  // Change here to use "context"
  });

  try {
    const response = await ragChain.invoke({
      question: query,
      context: formattedDocs  // Pass documents under "context"
    });
    
    return response;
  } catch (error) {
    console.error("Error during ragChain invocation:", error);
    throw error; // Rethrow the error for further handling if needed
  }
}




// Example usage
(async () => {
  const question = "What are the different types of heart failure?";
  const answer = await retrieveAndGenerate(question);
  console.log(answer);
})();
