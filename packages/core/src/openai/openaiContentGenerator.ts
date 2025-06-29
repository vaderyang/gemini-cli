/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
import {
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensResponse,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
  Part,
  FunctionCall,
  FunctionResponse,
  FinishReason,
  UsageMetadata,
} from '@google/genai';
import { ContentGenerator } from '../core/contentGenerator.js';

// Extended type for internal use only
type ThoughtPart = {
  thought: string;
};

type ExtendedPart = Part | ThoughtPart;

export class OpenAIContentGenerator implements ContentGenerator {
  private openai: OpenAI;

  constructor(
    apiKey: string | undefined,
    baseURL: string,
    httpOptions?: { headers?: Record<string, string> },
  ) {
    this.openai = new OpenAI({
      apiKey: apiKey || 'dummy-key', // OpenAI client requires a key even for local endpoints
      baseURL,
      defaultHeaders: httpOptions?.headers,
    });
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const stream = await this.generateContentStream(request);
    let finalResponse: GenerateContentResponse | undefined;
    
    for await (const response of stream) {
      finalResponse = response;
    }
    
    if (!finalResponse) {
      throw new Error('No response received from OpenAI API');
    }
    
    return finalResponse;
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const generator = this.doGenerateContentStream(request);
    return generator;
  }

  private async *doGenerateContentStream(
    request: GenerateContentParameters,
  ): AsyncGenerator<GenerateContentResponse> {
    const { model, config, contents } = request;
    
    // Convert to Content array
    const contentArray = this.normalizeContents(contents);
    
    // Convert Gemini format to OpenAI format
    const messages = this.convertContentsToOpenAIMessages(contentArray);
    
    // Add system instruction if present
    if (config?.systemInstruction) {
      const systemMessage = this.extractTextFromSystemInstruction(config.systemInstruction);
      if (systemMessage) {
        messages.unshift({ role: 'system', content: systemMessage });
      }
    }

    // Convert tools to OpenAI format
    let openAITools: OpenAI.Chat.ChatCompletionTool[] | undefined;
    if (config?.tools) {
      const toolsList: OpenAI.Chat.ChatCompletionTool[] = [];
      for (const tool of config.tools) {
        if ('functionDeclarations' in tool && tool.functionDeclarations) {
          for (const func of tool.functionDeclarations) {
            toolsList.push({
              type: 'function' as const,
              function: {
                name: func.name || 'unnamed',
                description: func.description || '',
                parameters: (func.parameters || {}) as any,
              }
            });
          }
        }
      }
      if (toolsList.length > 0) {
        openAITools = toolsList;
      }
    }

    try {
      const stream = await this.openai.chat.completions.create({
        model: model || 'qwen3-8b',
        messages,
        temperature: config?.temperature,
        top_p: config?.topP,
        max_tokens: config?.maxOutputTokens,
        tools: openAITools,
        stream: true,
      });

      let accumulatedText = '';
      let accumulatedThinking = '';
      let isInThinkingBlock = false;
      let functionCalls: FunctionCall[] = [];
      let currentToolCall: OpenAI.Chat.Completions.ChatCompletionChunk.Choice.Delta.ToolCall | null = null;

      // Track what has already been sent to avoid duplicating output
      let lastSentTextLength = 0;
      let lastSentFunctionCallCount = 0;

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta;
        
        if (!delta) continue;

        // Handle tool calls
        if (delta.tool_calls) {
          for (const toolCall of delta.tool_calls) {
            if (toolCall.index === 0 && toolCall.function?.name) {
              // Start of a new tool call
              if (currentToolCall && currentToolCall.function?.name && currentToolCall.function?.arguments) {
                // Save the previous tool call
                functionCalls.push({
                  name: currentToolCall.function.name,
                  args: JSON.parse(currentToolCall.function.arguments),
                });
              }
              currentToolCall = {
                index: toolCall.index,
                id: toolCall.id,
                type: 'function',
                function: {
                  name: toolCall.function.name,
                  arguments: toolCall.function.arguments || '',
                },
              };
            } else if (currentToolCall && toolCall.index === currentToolCall.index) {
              // Continue accumulating arguments for current tool call
              if (toolCall.function?.arguments) {
                currentToolCall.function!.arguments += toolCall.function.arguments;
              }
            }
          }
        }

        // Handle text content with <think></think> parsing
        if (delta.content) {
          const content = delta.content;
          
          // Check for thinking blocks
          if (!isInThinkingBlock && content.includes('<think>')) {
            const parts = content.split('<think>');
            accumulatedText += parts[0];
            isInThinkingBlock = true;
            accumulatedThinking = parts[1] || '';
          } else if (isInThinkingBlock && content.includes('</think>')) {
            const parts = content.split('</think>');
            accumulatedThinking += parts[0];
            isInThinkingBlock = false;
            // Continue with text after thinking block
            if (parts[1]) {
              accumulatedText += parts[1];
            }
          } else if (isInThinkingBlock) {
            accumulatedThinking += content;
          } else {
            accumulatedText += content;
          }
        }

        // Determine new text/function calls since last yield
        const newTextSegment = accumulatedText.slice(lastSentTextLength);
        const newFunctionCalls = functionCalls.slice(lastSentFunctionCallCount);

        // Update trackers
        if (newTextSegment) {
          lastSentTextLength = accumulatedText.length;
        }
        if (newFunctionCalls.length > 0) {
          lastSentFunctionCallCount = functionCalls.length;
        }

        // Only yield if there is new information
        if (newTextSegment || newFunctionCalls.length > 0) {
          const parts: Part[] = [];
          if (newTextSegment) {
            parts.push({ text: newTextSegment });
          }
          if (newFunctionCalls.length > 0) {
            newFunctionCalls.forEach(fc => parts.push({ functionCall: fc }));
          }

          const response = this.createResponse(
            parts,
            chunk.choices[0]?.finish_reason,
            accumulatedText, // Pass full text for usage estimation
            messages,
            accumulatedThinking
          );

          yield response;
        }
      }

      // Handle final tool call if exists
      if (currentToolCall && currentToolCall.function?.name && currentToolCall.function?.arguments) {
        functionCalls.push({
          name: currentToolCall.function.name,
          args: JSON.parse(currentToolCall.function.arguments),
        });
      }

      // Yield any function calls not yet sent (without re-sending text)
      const remainingFunctionCalls = functionCalls.slice(lastSentFunctionCallCount);
      if (remainingFunctionCalls.length > 0) {
        const parts: Part[] = [];
        remainingFunctionCalls.forEach(fc => parts.push({ functionCall: fc }));

        yield this.createResponse(
          parts,
          'stop',
          accumulatedText,
          messages,
          accumulatedThinking
        );
      }

      // The stream is finished; send a final empty chunk with finish_reason 'stop' if it wasn't already sent.
      yield this.createResponse(
        [],
        'stop',
        accumulatedText,
        messages,
        accumulatedThinking,
      );

    } catch (error) {
      throw new Error(`OpenAI API error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    // Estimate token count (OpenAI doesn't have a direct token counting endpoint)
    // This is a rough estimate: ~4 characters per token
    const contentArray = this.normalizeContents(request.contents);
    let totalChars = 0;
    
    for (const content of contentArray) {
      const text = this.contentToText(content);
      totalChars += text.length;
    }
    
    const estimatedTokens = Math.ceil(totalChars / 4);
    
    return {
      totalTokens: estimatedTokens,
    };
  }

  async embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse> {
    const { model, contents } = request;
    
    // Convert contents to strings
    const contentsList = Array.isArray(contents) ? contents : [contents];
    const inputs = contentsList.map(content => {
      if (typeof content === 'string') return content;
      if ('parts' in content) return this.contentToText(content as Content);
      return ''; // Part objects are not valid for embeddings
    });
    
    try {
      const response = await this.openai.embeddings.create({
        model: model || 'text-embedding-3-small',
        input: inputs,
      });
      
      return {
        embeddings: response.data.map(item => ({
          values: item.embedding,
        })),
      };
    } catch (error) {
      throw new Error(`OpenAI Embeddings API error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private createResponse(
    parts: Part[], 
    finishReason: string | null | undefined,
    text: string,
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
    thinking?: string
  ): GenerateContentResponse {
    // Create a response object that matches the expected interface
    const response = {
      candidates: [{
        content: {
          role: 'model' as const,
          parts,
        },
        finishReason: this.mapFinishReason(finishReason),
        safetyRatings: [],
        index: 0,
      }],
      usageMetadata: this.estimateUsageMetadata(text, messages),
      // Getter properties
      get text() { return parts.find(p => 'text' in p)?.text || ''; },
      get data() { return undefined; },
      get functionCalls() { return parts.filter(p => 'functionCall' in p).map(p => (p as any).functionCall as FunctionCall); },
      get executableCode() { return undefined; },
      get codeExecutionResult() { return undefined; },
    } as unknown as GenerateContentResponse;

    // Store thinking data separately if needed
    if (thinking) {
      // We can store this in a custom property if needed by the application
      (response as any)._thinking = thinking;
    }

    return response;
  }

  private mapFinishReason(reason: string | null | undefined): FinishReason {
    // Use enum values directly
    switch (reason) {
      case 'stop':
        return 'STOP' as FinishReason;
      case 'length':
        return 'MAX_TOKENS' as FinishReason;
      case 'function_call':
      case 'tool_calls':
        return 'STOP' as FinishReason;
      case 'content_filter':
        return 'SAFETY' as FinishReason;
      default:
        return 'OTHER' as FinishReason;
    }
  }

  private normalizeContents(contents: any): Content[] {
    if (Array.isArray(contents)) {
      return contents.map(c => {
        if (typeof c === 'string') {
          return { role: 'user', parts: [{ text: c }] };
        }
        if (c && typeof c === 'object' && 'parts' in c) {
          return c as Content;
        }
        // If it's a Part or Part[], wrap it in a Content
        const parts = Array.isArray(c) ? c : [c];
        return { role: 'user', parts };
      });
    }
    
    if (typeof contents === 'string') {
      return [{ role: 'user', parts: [{ text: contents }] }];
    }
    
    if (contents && typeof contents === 'object' && 'parts' in contents) {
      return [contents as Content];
    }
    
    // Assume it's a Part or Part[]
    const parts = Array.isArray(contents) ? contents : [contents];
    return [{ role: 'user', parts }];
  }

  private extractTextFromSystemInstruction(instruction: any): string {
    if (typeof instruction === 'string') return instruction;
    if (Array.isArray(instruction)) {
      return instruction.map(p => {
        if (typeof p === 'string') return p;
        if (p && 'text' in p) return p.text;
        return '';
      }).join('');
    }
    if (instruction && 'text' in instruction) return instruction.text;
    return '';
  }

  private convertContentsToOpenAIMessages(contents: Content[]): OpenAI.Chat.ChatCompletionMessageParam[] {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    
    for (const content of contents) {
      const role = content.role === 'model' ? 'assistant' : content.role;
      const parts = content.parts || [];
      const text = this.partsToText(parts);
      
      // Check for function calls in parts
      const functionCalls = parts.filter(part => 'functionCall' in part);
      const functionResponses = parts.filter(part => 'functionResponse' in part);
      
      if (functionCalls.length > 0) {
        // Convert to assistant message with tool calls
        const toolCalls = functionCalls.map((part, index) => {
          const fc = (part as any).functionCall as FunctionCall;
          return {
            id: `call_${index}`,
            type: 'function' as const,
            function: {
              name: fc.name || 'unknown',
              arguments: JSON.stringify(fc.args),
            },
          };
        });
        
        messages.push({
          role: 'assistant' as const,
          content: text || null,
          tool_calls: toolCalls,
        });
      } else if (functionResponses.length > 0) {
        // Convert function responses to tool messages
        functionResponses.forEach(part => {
          const fr = (part as any).functionResponse as FunctionResponse;
          messages.push({
            role: 'tool' as const,
            content: JSON.stringify(fr.response),
            tool_call_id: `call_${fr.name}`, // This is a simplification
          });
        });
      } else {
        messages.push({
          role: role as 'user' | 'assistant' | 'system',
          content: text || '',
        });
      }
    }
    
    return messages;
  }

  private partsToText(parts: Part[] | undefined): string {
    if (!parts) return '';
    return parts
      .map(part => {
        if ('text' in part) return part.text || '';
        return '';
      })
      .join('');
  }

  private contentToText(content: Content): string {
    return this.partsToText(content.parts);
  }

  private estimateUsageMetadata(text: string, messages: OpenAI.Chat.ChatCompletionMessageParam[]): UsageMetadata {
    // Rough token estimation
    const promptChars = messages.reduce((sum, msg) => 
      sum + (typeof msg.content === 'string' ? msg.content.length : 0), 0
    );
    const promptTokens = Math.ceil(promptChars / 4);
    const candidatesTokens = Math.ceil(text.length / 4);
    
    return {
      promptTokenCount: promptTokens,
      totalTokenCount: promptTokens + candidatesTokens,
    };
  }
}