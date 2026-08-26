import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

function usage() {
  return [
    "Usage: node deliver_mopso_calibration_visual_report.mjs",
    "  <build-report-scripts-dir> <artifact.json> <report.html>",
  ].join("\n");
}

const [, , scriptsDirectory, inputPath, outputPath] = process.argv;
if (!scriptsDirectory || !inputPath || !outputPath) {
  throw new Error(usage());
}

const importFromScripts = (name) =>
  import(pathToFileURL(resolve(scriptsDirectory, name)).href);

const [{ buildPortableArtifact }, { deliverPortableArtifact }] = await Promise.all([
  importFromScripts("build_portable_artifact.mjs"),
  importFromScripts("deliver_portable_artifact.mjs"),
]);

// The portable reader header uses 100vw. On Windows, a long report's vertical
// scrollbar makes that a few pixels wider than the document viewport. Clipping
// only that harmless horizontal spill preserves all report content and lets the
// packaged desktop/mobile verification exercise the actual charts.
function buildWithoutScrollbarSpill(artifact, options) {
  const html = buildPortableArtifact(artifact, options);
  const style = "<style data-mopso-overflow-fix>html,body{overflow-x:clip}</style>";
  return html.replace("</head>", `${style}</head>`);
}

const result = await deliverPortableArtifact(
  {
    actionTimeoutMs: 5_000,
    inputPath: resolve(inputPath),
    outputPath: resolve(outputPath),
    readyTimeoutMs: 20_000,
    timeoutMs: 40_000,
  },
  { build: buildWithoutScrollbarSpill },
);

process.stdout.write(`${JSON.stringify(result)}\n`);
