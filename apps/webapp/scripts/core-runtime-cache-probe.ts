import { makeTextModelCall } from "../app/lib/model.server";

function repeatText(seed: string, size: number): string {
  return seed.repeat(Math.ceil(size / seed.length)).slice(0, size);
}

async function main() {
  const staticPadding = Number(
    process.env.CORE_RUNTIME_CACHE_PAD_CHARS || "60000",
  );
  const userText =
    process.env.CORE_RUNTIME_CACHE_USER_TEXT || 'Reply with exactly "ok".';
  const staticSystem = [
    "You are a cache verification probe running through CORE's real model stack.",
    "Return exactly the requested answer and nothing else.",
    "",
    "<static_block>",
    repeatText("stable-cache-padding ", staticPadding),
    "</static_block>",
  ].join("\n");

  const messages = [
    { role: "system" as const, content: staticSystem },
    { role: "user" as const, content: userText },
  ];

  const first = await makeTextModelCall(
    messages,
    {
      maxOutputTokens: 16,
      temperature: 0,
    },
    "high",
    "core-runtime-cache-probe",
    "low",
    { callSite: "core.runtime.cache.probe" },
  );

  const second = await makeTextModelCall(
    messages,
    {
      maxOutputTokens: 16,
      temperature: 0,
    },
    "high",
    "core-runtime-cache-probe",
    "low",
    { callSite: "core.runtime.cache.probe" },
  );

  console.log(
    JSON.stringify(
      {
        first: first.usage,
        second: second.usage,
        firstText: first.text,
        secondText: second.text,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
