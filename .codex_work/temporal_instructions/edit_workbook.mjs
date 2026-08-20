import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/macbooka/Downloads/maniskill_instructions_to_label.xlsx";
const outputDir = "/Users/macbooka/Documents/VsCode/multimodal/outputs/temporal_instructions_20260819";
const workDir = "/Users/macbooka/Documents/VsCode/multimodal/.codex_work/temporal_instructions";
const outputPath = `${outputDir}/maniskill_instructions_with_temporal.xlsx`;

const temporalRows = [
  [10, "pick up the object that was moved most recently.", null, null, "temporal", null, "high", "Synthetic temporal-memory source task; requires identifying the most recently moved object.", "train"],
  [11, "return the red cube to the position it occupied before it was moved.", null, null, "temporal", null, "high", "Synthetic temporal-memory source task; requires recalling an earlier object position.", "train"],
  [12, "place the mug where it was last seen.", null, null, "temporal", null, "high", "Synthetic temporal-memory source task; requires retrieving the latest prior observation of the mug.", "train"],
  [13, "retrieve the object that was handled immediately before the green cube.", null, null, "temporal", null, "high", "Synthetic temporal-memory source task; requires reasoning over action order.", "train"],
  [14, "restore the faucet handle to its previous state.", null, null, "temporal", null, "high", "Synthetic temporal-memory source task; requires recalling the faucet handle's prior state.", "train"],
  [15, "repeat the most recent successful pick-and-place action.", null, null, "temporal", null, "high", "Synthetic temporal-memory source task; requires retrieving the latest successful action.", "train"],
  [16, "move the block back to its initial position after completing the placement.", null, null, "temporal", null, "high", "Synthetic temporal-memory source task; requires remembering the block's initial position.", "train"],
];

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItemAt(0);
const tableHelp = workbook.help("worksheet.tables", {
  include: "index,examples,notes",
  maxChars: 5000,
});
console.log(tableHelp.ndjson);

const primaryModes = sheet.getRange("E2:E10").values.map(([value]) => [value === "spatial" ? "navigation" : value]);
sheet.getRange("E2:E10").values = primaryModes;

sheet.getRange("A11:I17").values = temporalRows;

sheet.freezePanes.freezeRows(1);
sheet.getRange("A1:I17").format.font = { name: "Arial", size: 10 };
sheet.getRange("A1:I1").format = {
  fill: "#17365D",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  rowHeight: 28,
  verticalAlignment: "center",
};
sheet.getRange("A2:I17").format.verticalAlignment = "center";
sheet.getRange("B2:B17").format.wrapText = true;
sheet.getRange("H2:H17").format.wrapText = true;
sheet.getRange("A:A").format.columnWidth = 15;
sheet.getRange("B:B").format.columnWidth = 62;
sheet.getRange("C:D").format.columnWidth = 15;
sheet.getRange("E:F").format.columnWidth = 18;
sheet.getRange("G:G").format.columnWidth = 12;
sheet.getRange("H:H").format.columnWidth = 62;
sheet.getRange("I:I").format.columnWidth = 12;
sheet.getRange("A2:A17").format.numberFormat = "0";
sheet.getRange("C2:D17").format.numberFormat = "#,##0";
sheet.getRange("10:17").format.rowHeight = 34;

sheet.getRange("E2:E17").dataValidation = {
  rule: { type: "list", values: ["navigation", "object_state", "default", "temporal"] },
};
sheet.getRange("G2:G17").dataValidation = {
  rule: { type: "list", values: ["high", "medium", "low"] },
};
sheet.getRange("I2:I17").dataValidation = {
  rule: { type: "list", values: ["train", "validation", "test"] },
};

const check = await workbook.inspect({
  kind: "table",
  range: `'${sheet.name}'!A1:I17`,
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 10,
  maxChars: 16000,
});
console.log(check.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: sheet.name,
  range: "A1:I17",
  scale: 1.25,
  format: "png",
});
await fs.writeFile(`${workDir}/output_preview.png`, new Uint8Array(await preview.arrayBuffer()));

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`output=${outputPath}`);
