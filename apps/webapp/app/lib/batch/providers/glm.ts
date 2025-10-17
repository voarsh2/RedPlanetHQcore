import OpenAI from "openai";
import { BaseBatchProvider } from "../base-provider";
import {
  type CreateBatchParams,
  type GetBatchParams,
  type BatchJob,
  type BatchResponse,
  type BatchStatus,
} from "../types";
import { logger } from "~/services/logger.service";
import { getModelForTask } from "~/lib/model.server";

interface StoredBatch {
  job: BatchJob;
  cancelRequested: boolean;
}

export class GLMBatchProvider extends BaseBatchProvider {
  providerName = "glm";
  supportedModels = ["glm-4", "glm-4-air", "glm-4-airx", "glm-4-flash", "glm-*"];

  private static batches: Map<string, StoredBatch> = new Map();

  private glmClient: OpenAI;

  constructor(options?: { apiKey?: string; baseURL?: string }) {
    super();

    const apiKey = options?.apiKey || process.env.GLM_API_KEY;
    if (!apiKey) {
      throw new Error("GLM_API_KEY environment variable is required for GLM batch processing");
    }

    this.glmClient = new OpenAI({
      apiKey,
      baseURL: options?.baseURL || process.env.GLM_API_BASE_URL || "https://open.bigmodel.cn/api/paas/v4",
    });
  }

  async createBatch<T>(params: CreateBatchParams<T>): Promise<{ batchId: string }> {
    this.validateRequests(params.requests);

    const model = getModelForTask(params.modelComplexity || "high");
    if (!this.isModelSupported(model)) {
      throw new Error(`Model ${model} is not supported by the GLM batch provider`);
    }

    const batchId = this.generateBatchId();
    const createdAt = new Date();

    const initialJob: BatchJob = {
      batchId,
      status: "processing",
      totalRequests: params.requests.length,
      completedRequests: 0,
      failedRequests: 0,
      createdAt,
      results: [],
    };

    GLMBatchProvider.batches.set(batchId, {
      job: initialJob,
      cancelRequested: false,
    });

    void this.processBatch(batchId, model, params);

    logger.info(`GLM batch created: ${batchId}`);
    return { batchId };
  }

  async getBatch<T>(params: GetBatchParams): Promise<BatchJob> {
    const stored = GLMBatchProvider.batches.get(params.batchId);
    if (!stored) {
      throw new Error(`Batch ${params.batchId} not found`);
    }

    return stored.job;
  }

  async cancelBatch(params: GetBatchParams): Promise<{ success: boolean }> {
    const stored = GLMBatchProvider.batches.get(params.batchId);
    if (!stored) {
      return { success: false };
    }

    if (stored.job.status !== "processing") {
      return { success: false };
    }

    stored.cancelRequested = true;
    stored.job.status = "cancelled";
    stored.job.completedAt = new Date();
    stored.job.results = stored.job.results || [];
    GLMBatchProvider.batches.set(params.batchId, stored);
    logger.info(`GLM batch cancelled: ${params.batchId}`);
    return { success: true };
  }

  private async processBatch<T>(
    batchId: string,
    model: string,
    params: CreateBatchParams<T>,
  ): Promise<void> {
    const stored = GLMBatchProvider.batches.get(batchId);
    if (!stored) {
      return;
    }

    const job = stored.job;
    const results: BatchResponse[] = [];
    let completed = 0;
    let failed = 0;

    for (const request of params.requests) {
      const latest = GLMBatchProvider.batches.get(batchId);
      if (!latest) {
        return;
      }

      if (latest.cancelRequested) {
        logger.info(`GLM batch processing stopped due to cancellation: ${batchId}`);
        job.status = "cancelled";
        job.completedAt = new Date();
        job.results = results;
        latest.job = job;
        GLMBatchProvider.batches.set(batchId, latest);
        return;
      }

      try {
        const messages = request.systemPrompt
          ? [
              { role: "system" as const, content: request.systemPrompt },
              ...request.messages,
            ]
          : request.messages;

        const completion = await this.glmClient.chat.completions.create({
          model,
          messages,
          ...request.options,
          ...(params.outputSchema && {
            response_format: {
              type: "json_schema" as const,
              json_schema: {
                name: "structured_output",
                strict: true,
                schema: this.zodToJsonSchema(params.outputSchema),
              },
            },
          }),
        });

        const content = completion.choices?.[0]?.message?.content ?? "";

        results.push({
          customId: request.customId,
          response: content,
        });
        completed += 1;
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Unknown error occurred during GLM batch processing";

        logger.error("GLM batch request failed:", { batchId, error: message });

        results.push({
          customId: request.customId,
          error: {
            code: "glm_api_error",
            message,
            type: "api_error",
          },
        });
        failed += 1;
      }

      job.completedRequests = completed;
      job.failedRequests = failed;
      job.results = [...results];
      stored.job = job;
      GLMBatchProvider.batches.set(batchId, stored);
    }

    if (!stored.cancelRequested) {
      job.status = failed === params.requests.length ? "failed" : "completed";
      job.completedAt = new Date();
      job.results = results;
      stored.job = job;
      GLMBatchProvider.batches.set(batchId, stored);
      logger.info(`GLM batch processing completed: ${batchId}`);
    }
  }

  private zodToJsonSchema(schema: any): any {
    if (!schema?._def) {
      return { type: "string" };
    }

    switch (schema._def.typeName) {
      case "ZodObject":
        return {
          type: "object",
          properties: Object.fromEntries(
            Object.entries(schema._def.shape()).map(([key, value]: [string, any]) => [
              key,
              this.zodToJsonSchema(value),
            ]),
          ),
          required: Object.entries(schema._def.shape())
            .filter(([, value]: [string, any]) => !value.isOptional?.())
            .map(([key]) => key),
          additionalProperties: false,
        };
      case "ZodArray":
        return {
          type: "array",
          items: this.zodToJsonSchema(schema._def.type),
        };
      case "ZodString":
        return { type: "string" };
      case "ZodNumber":
        return { type: "number" };
      case "ZodBoolean":
        return { type: "boolean" };
      default:
        return { type: "string" };
    }
  }
}
