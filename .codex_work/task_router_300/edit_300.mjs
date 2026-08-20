import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/macbooka/Desktop/maniskill_router_paraphrases_270.xlsx";
const outputDir = "/Users/macbooka/Documents/VsCode/multimodal/outputs/task_router_300_20260819";
const workDir = "/Users/macbooka/Documents/VsCode/multimodal/.codex_work/task_router_300";
const outputPath = `${outputDir}/maniskill_router_paraphrases_300.xlsx`;

const temporalInstructions = [
  "return the red cube to the position it occupied before it was moved.",
  "move the red cube back to its previous position.",
  "place the red cube at the location it occupied before its last move.",
  "restore the red cube to its prior location.",
  "take the red cube back to where it was previously.",
  "relocate the red cube to the position it held before being moved.",
  "put the red cube back in its earlier location.",
  "return the red block to the spot it occupied immediately before the current one.",
  "move the red cube from its current location back to its preceding position.",
  "restore the red block to its last position before relocation.",
  "place the cube colored red where it was before the most recent move.",
  "take the red block back to its previous recorded location.",
  "move the red cube back to the location it came from.",
  "return the red block to its earlier position before the transfer.",
  "reposition the red cube at the location it occupied one move ago.",
  "put the red cube back where it had been before.",
  "restore the red cube to the position preceding its current position.",
  "move the red block back to its last known position before relocation.",
  "return the red cube to the location from which it was moved.",
  "place the red cube in the prior position remembered from before the move.",
  "undo the most recent move by returning the red cube to its earlier location.",
  "reverse the red cube's last relocation and restore its prior position.",
  "move the red cube back to the spot it occupied previously.",
  "return the red block to the location it was in before its latest movement.",
  "restore the red cube to the exact position it had before it moved.",
  "place the red cube back at its former location.",
  "take the red cube to its immediately previous position.",
  "return the red block to the starting point of its most recent move.",
  "move the red cube back to where it was just before relocation.",
  "restore the red cube to the position it occupied prior to its current location.",
];

if (temporalInstructions.length !== 30) throw new Error("Expected exactly 30 temporal instructions");
const normalizedTemporal = new Set(temporalInstructions.map((text) => text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim()));
if (normalizedTemporal.size !== 30) throw new Error("Temporal paraphrases must be unique");

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const source = workbook.worksheets.getItem("Source Tasks");
const readme = workbook.worksheets.getItem("README");
const augmented = workbook.worksheets.getItem("Augmented Instructions");
const summary = workbook.worksheets.getItem("Class Summary");

source.getRange("E2:E10").values = source.getRange("E2:E10").values.map(([value]) => [value === "spatial" ? "navigation" : value]);
const temporalNote = "Synthetic temporal-memory source task; requires recall of the red cube's immediately previous position.";
source.getRange("A11:I11").values = [[10, temporalInstructions[0], null, null, "temporal", "", "high", temporalNote, "train"]];
source.getRange("A11:I11").format = { font: { name: "Aptos", size: 10 }, verticalAlignment: "center" };
source.getRange("A11:I11").format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" } };
source.getRange("B11:H11").format.wrapText = true;
source.getRange("11:11").format.rowHeight = 38;
source.getRange("E2:E11").dataValidation = { rule: { type: "list", values: ["navigation", "object_state", "default", "temporal"] } };

readme.getRange("B3").values = [[10]];
readme.getRange("B5").values = [[300]];
readme.getRange("B6").values = [["Inherited exactly from Source Tasks: navigation, object_state, default, temporal"]];
readme.getRange("B7").values = [["All rows are train. Do not validate on paraphrases of these same ten source tasks."]];
readme.getRange("B9").values = [["Supported by one synthetic source task with 30 reviewed paraphrases requiring recall of a previous object position."]];
readme.getRange("B12").values = [["300 rows represent only 10 independent task meanings; paraphrasing adds wording diversity, not task diversity."]];

augmented.getRange("F2:F271").values = augmented.getRange("F2:F271").values.map(([value]) => [value === "spatial" ? "navigation" : value]);
const newRows = temporalInstructions.map((instruction, index) => [
  271 + index,
  10,
  "maniskill_task_10",
  instruction,
  index === 0,
  "temporal",
  "",
  index === 0 ? "source_label" : "inherited",
  index === 0 ? "source" : "assistant_reviewed",
  "train",
  null,
  null,
  null,
  temporalNote,
]);
augmented.getRange("A272:N301").values = newRows;
augmented.getRange("M2").formulas = [["=IF(F2=\"navigation\",'Class Summary'!$D$2,IF(F2=\"object_state\",'Class Summary'!$D$3,IF(F2=\"default\",'Class Summary'!$D$4,IF(F2=\"temporal\",'Class Summary'!$D$5,\"\"))))"]];
augmented.getRange("M2:M301").fillDown();
augmented.getRange("A272:N301").format = { font: { name: "Aptos", size: 10 }, verticalAlignment: "center" };
augmented.getRange("A272:N301").format.borders = { insideHorizontal: { style: "thin", color: "#E7EEF5" } };
augmented.getRange("D272:D301").format.wrapText = true;
augmented.getRange("N272:N301").format.wrapText = true;
augmented.getRange("K272:L301").format.numberFormat = "#,##0";
augmented.getRange("M2:M301").format.numberFormat = "0.000";
augmented.getRange("F2:F301").dataValidation = { rule: { type: "list", values: ["navigation", "object_state", "default", "temporal"] } };
augmented.getRange("I2:I301").dataValidation = { rule: { type: "list", values: ["source", "assistant_reviewed", "needs_review"] } };

summary.getRange("A2").values = [["navigation"]];
summary.getRange("B2").formulas = [["=COUNTIF('Augmented Instructions'!$F$2:$F$301,A2)"]];
summary.getRange("B2:B5").fillDown();

const classCheck = await workbook.inspect({
  kind: "table",
  range: "'Class Summary'!A1:D8",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 6,
  maxChars: 8000,
});
console.log(classCheck.ndjson);

const temporalCheck = await workbook.inspect({
  kind: "table",
  range: "'Augmented Instructions'!A268:N301",
  include: "values,formulas",
  tableMaxRows: 36,
  tableMaxCols: 14,
  maxChars: 22000,
});
console.log(temporalCheck.ndjson);

const structureCheck = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 10000,
  tableMaxRows: 4,
  tableMaxCols: 6,
});
console.log(structureCheck.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const previews = [
  ["Source Tasks", "A1:I11", "output_source_tasks.png"],
  ["README", "A1:B12", "output_readme.png"],
  ["Augmented Instructions", null, "output_augmented_instructions.png"],
  ["Augmented Instructions", "A268:N301", "output_temporal_rows.png"],
  ["Class Summary", "A1:D8", "output_class_summary.png"],
];
for (const [sheetName, range, fileName] of previews) {
  const options = { sheetName, scale: range ? 1.25 : 0.75, format: "png" };
  if (range) options.range = range;
  else options.autoCrop = "all";
  const preview = await workbook.render(options);
  await fs.writeFile(`${workDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`output=${outputPath}`);
