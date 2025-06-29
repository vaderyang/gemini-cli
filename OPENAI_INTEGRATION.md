# OpenAI API Integration for Gemini CLI

This document describes how to use the OpenAI API integration with Gemini CLI, including support for local OpenAI-compatible endpoints.

## Setup

### 1. Environment Variables

Set the following environment variables in your `.env` file:

```bash
# For OpenAI API
OPENAI_API_KEY=your-openai-api-key

# For custom endpoint (e.g., local LLM server)
OPENAI_BASE_URL=http://localhost:1234/v1
```

Note: If using a local endpoint that doesn't require authentication, you can set any dummy value for `OPENAI_API_KEY`.

### 2. Select OpenAI Authentication

When you run the Gemini CLI, select "OpenAI API" from the authentication options:

```bash
gemini
```

Then choose option 4: "OpenAI API"

## Features

### 1. Custom Base URL Support

The integration supports custom OpenAI-compatible endpoints, making it work with:
- Local LLM servers (LM Studio, Ollama with OpenAI compatibility layer, etc.)
- OpenAI API proxies
- Alternative OpenAI-compatible services

### 2. Thinking Block Support

The integration handles `<think></think>` tags in responses, which some local models use for reasoning:

```
User: What is 2+2?

Model: <think>The user is asking a simple arithmetic question...</think>The answer is 4.
```

The thinking content is parsed and stored separately in the response, maintaining compatibility with Gemini's thinking feature.

### 3. Tool Calling

Full support for OpenAI's function calling format, automatically converted to Gemini's tool format:

```javascript
// Tools are automatically converted from Gemini format to OpenAI format
const tools = [{
  functionDeclarations: [{
    name: 'get_weather',
    description: 'Get weather information',
    parameters: {
      type: 'object',
      properties: {
        location: { type: 'string' }
      }
    }
  }]
}];
```

### 4. Model Configuration

Default model is `gpt-4`, but you can specify any model supported by your endpoint:

```bash
gemini --model gpt-3.5-turbo
# or for local models
gemini --model mistral-7b-instruct
```

### 5. Embeddings

When using OpenAI authentication, the CLI automatically uses OpenAI's embedding model (`text-embedding-3-small`) instead of Gemini's.

## Example Usage

### Basic Chat
```bash
gemini "Hello, how are you?"
```

### With Local Endpoint
```bash
export OPENAI_BASE_URL=http://localhost:1234/v1
gemini "Explain quantum computing"
```

### Tool Usage
The existing Gemini CLI tools work seamlessly:
```bash
gemini "Read the README.md file and summarize it"
```

## Testing the Integration

### Quick Test with the CLI

1. Set up environment variables:
```bash
export OPENAI_BASE_URL=http://localhost:1234/v1
export OPENAI_API_KEY=dummy-key  # or your actual OpenAI API key
```

2. Run the CLI:
```bash
# From the project root
node bundle/gemini.js
# Or if you have it installed globally
gemini
```

3. Select "OpenAI API" when prompted for authentication method

4. Test with a simple query:
```bash
gemini "Hello, can you hear me?"
```

### Integration Test Script

A test script is provided to verify the integration:

```bash
# Build the project first
npm run build

# Run the test
node test-openai-integration.js
```

## Troubleshooting

1. **Connection Refused**: Ensure your local endpoint is running and accessible at the specified URL
2. **Authentication Error**: Check that your API key is correct (or use a dummy key for local endpoints)
3. **Model Not Found**: Verify the model name is supported by your endpoint
4. **Tool Calling Issues**: Some local models may not support function calling - check your model's capabilities

## Implementation Details

The OpenAI integration is implemented through:
- `OpenAIContentGenerator` class that implements the `ContentGenerator` interface
- Automatic conversion between OpenAI and Gemini message formats
- Streaming support with proper chunk handling
- Token counting estimation (as OpenAI doesn't provide a direct token counting endpoint)