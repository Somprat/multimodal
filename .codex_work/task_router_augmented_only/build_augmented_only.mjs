import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputPath = "/Users/macbooka/Desktop/maniskill_router_paraphrases_300.xlsx";
const outputDir = "/Users/macbooka/Documents/VsCode/multimodal/outputs/task_router_augmented_only_20260819";
const workDir = "/Users/macbooka/Documents/VsCode/multimodal/.codex_work/task_router_augmented_only";
const outputPath = `${outputDir}/maniskill_augmented_instructions_300.xlsx`;

const input = await FileBlob.load(inputPath);
const sourceWorkbook = await SpreadsheetFile.importXlsx(input);
const sourceSheet = sourceWorkbook.worksheets.getItem("Augmented Instructions");
const sourceValues = sourceSheet.getRange("A1:N301").values;

if (sourceValues.length !== 301 || sourceValues[0].length !== 14) {
  throw new Error(`Unexpected source shape: ${sourceValues.length}x${sourceValues[0]?.length ?? 0}`);
}

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Augmented Instructions");
sheet.getRange("A1:N301").values = sourceValues;
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(3);

sheet.getRange("A1:N1").format = {
  fill: "#17365D",
  font: { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" },
  rowHeight: 28,
  wrapText: true,
};
sheet.getRange("A2:N301").format = { font: { name: "Aptos", size: 10 }, verticalAlignment: "center" };
sheet.getRange("A2:N301").format.borders = { insideHorizontal: { style: "thin", color: "#E7EEF5" } };
sheet.getRange("A:A").format.columnWidth = 16;
sheet.getRange("B:B").format.columnWidth = 20;
sheet.getRange("C:C").format.columnWidth = 20;
sheet.getRange("D:D").format.columnWidth = 78;
sheet.getRange("E:E").format.columnWidth = 12;
sheet.getRange("F:G").format.columnWidth = 18;
sheet.getRange("H:I").format.columnWidth = 20;
sheet.getRange("J:J").format.columnWidth = 10;
sheet.getRange("K:L").format.columnWidth = 15;
sheet.getRange("M:M").format.columnWidth = 15;
sheet.getRange("N:N").format.columnWidth = 68;
sheet.getRange("D2:D301").format.wrapText = true;
sheet.getRange("N2:N301").format.wrapText = true;
sheet.getRange("K2:L301").format.numberFormat = "#,##0";
sheet.getRange("M2:M301").format.numberFormat = "0.000";
sheet.getRange("F2:F301").dataValidation = { rule: { type: "list", values: ["navigation", "object_state", "default", "temporal"] } };
sheet.getRange("I2:I301").dataValidation = { rule: { type: "list", values: ["source", "assistant_reviewed", "needs_review"] } };

const table = sheet.tables.add("A1:N301", true, "AugmentedInstructionsTable");
table.style = "TableStyleMedium2";
table.showFilterButton = true;

const labelCounts = {};
for (const row of sourceValues.slice(1)) {
  const label = row[5];
  labelCounts[label] = (labelCounts[label] ?? 0) + 1;
  if (typeof row[12] !== "number") throw new Error(`Non-numeric sample weight at augmentation_id ${row[0]}`);
}
console.log(JSON.stringify({ rows: sourceValues.length - 1, labelCounts }));

const topCheck = await workbook.inspect({
  kind: "table",
  range: "'Augmented Instructions'!A1:N12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 14,
  maxChars: 14000,
});
console.log(topCheck.ndjson);

const bottomCheck = await workbook.inspect({
  kind: "table",
  range: "'Augmented Instructions'!A268:N301",
  include: "values,formulas",
  tableMaxRows: 36,
  tableMaxCols: 14,
  maxChars: 22000,
});
console.log(bottomCheck.ndjson);

const formulaCheck = await workbook.inspect({
  kind: "formula",
  sheetId: "Augmented Instructions",
  range: "A1:N301",
  maxChars: 3000,
  options: { maxResults: 20 },
});
console.log(formulaCheck.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const fullPreview = await workbook.render({
  sheetName: "Augmented Instructions",
  autoCrop: "all",
  scale: 0.75,
  format: "png",
});
await fs.writeFile(`${workDir}/output_full.png`, new Uint8Array(await fullPreview.arrayBuffer()));

const bottomPreview = await workbook.render({
  sheetName: "Augmented Instructions",
  range: "A268:N301",
  scale: 1.25,
  format: "png",
});
await fs.writeFile(`${workDir}/output_bottom.png`, new Uint8Array(await bottomPreview.arrayBuffer()));

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`output=${outputPath}`);
