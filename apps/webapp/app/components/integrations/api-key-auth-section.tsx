import React, { useState, useCallback } from "react";
import { useFetcher } from "@remix-run/react";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";

interface ApiKeyAuthSectionProps {
  integration: {
    id: string;
    name: string;
  };
  specData: any;
  activeAccount: any;
}

export function ApiKeyAuthSection({
  integration,
  specData,
  activeAccount,
}: ApiKeyAuthSectionProps) {
  const apiKeyConfig = specData?.auth?.api_key;
  const [apiKey, setApiKey] = useState("");
  const [fieldValues, setFieldValues] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [showApiKeyForm, setShowApiKeyForm] = useState(false);
  const apiKeyFetcher = useFetcher();

  const fields: Array<{ name: string; label: string; placeholder?: string; description?: string }> | undefined =
    apiKeyConfig?.fields;

  const isMultiField = Array.isArray(fields) && fields.length > 0;
  const allowEmpty = !!apiKeyConfig?.allowEmpty && !isMultiField;

  const isSubmitDisabled = isMultiField
    ? fields.some((f) => !fieldValues[f.name]?.trim())
    : !allowEmpty && !apiKey.trim();

  const handleApiKeyConnect = useCallback(() => {
    if (isSubmitDisabled) return;

    setIsLoading(true);

    // For multi-field auth, send fields separately; for single-field, just send apiKey
    apiKeyFetcher.submit({
      integrationDefinitionId: integration.id,
      apiKey: isMultiField ? "" : apiKey,
      fields: isMultiField ? fieldValues : {},
    }, {
      method: "post",
      action: "/api/v1/integration_account",
      encType: "application/json",
    });
  }, [integration.id, apiKey, fieldValues, isMultiField, isSubmitDisabled, apiKeyFetcher]);

  React.useEffect(() => {
    if (apiKeyFetcher.state === "idle" && isLoading) {
      if (apiKeyFetcher.data !== undefined) {
        window.location.reload();
      }
    }
  }, [apiKeyFetcher.state, apiKeyFetcher.data, isLoading]);

  if (activeAccount || !apiKeyConfig) {
    return null;
  }

  return (
    <div className="bg-background-3 space-y-4 rounded-lg p-4">
      <h4 className="font-medium">API Key Authentication</h4>
      {!showApiKeyForm ? (
        <Button
          variant="secondary"
          onClick={() => setShowApiKeyForm(true)}
          className="w-full"
        >
          {apiKeyConfig?.connectLabel || "Connect with API Key"}
        </Button>
      ) : (
        <div className="flex flex-col gap-2">
          {isMultiField ? (
            fields.map((field) => (
              <div key={field.name} className="flex flex-col gap-1">
                <label htmlFor={field.name} className="text-sm font-medium">
                  {field.label}
                </label>
                <Input
                  id={field.name}
                  placeholder={field.placeholder || `Enter ${field.label}`}
                  value={fieldValues[field.name] ?? ""}
                  onChange={(e) =>
                    setFieldValues((prev) => ({
                      ...prev,
                      [field.name]: e.target.value,
                    }))
                  }
                />
                {field.description && (
                  <p className="text-muted-foreground text-sm">
                    {field.description}
                  </p>
                )}
              </div>
            ))
          ) : allowEmpty ? (
            <div className="flex flex-col gap-1">
              {apiKeyConfig?.description && (
                <p className="text-muted-foreground text-sm">
                  {apiKeyConfig.description}
                </p>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              <label htmlFor="apiKey" className="text-sm font-medium">
                {apiKeyConfig?.label || "API Key"}
              </label>
              <Input
                id="apiKey"
                placeholder="Enter your API key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
              {apiKeyConfig?.description && (
                <p className="text-muted-foreground text-sm">
                  {apiKeyConfig.description}
                </p>
              )}
            </div>
          )}
          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="ghost"
              onClick={() => {
                setShowApiKeyForm(false);
                setApiKey("");
                setFieldValues({});
              }}
            >
              Cancel
            </Button>
            <Button
              type="button"
              variant="default"
              disabled={isLoading || isSubmitDisabled}
              onClick={handleApiKeyConnect}
            >
              {isLoading || apiKeyFetcher.state === "submitting"
                ? "Connecting..."
                : "Connect"}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
