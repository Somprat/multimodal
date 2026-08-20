import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/macbooka/Downloads/maniskill_instructions_to_label.xlsx";
const workDir = "/Users/macbooka/Documents/VsCode/multimodal/.codex_work/task_router_paraphrases";
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 10000,
  tableMaxRows: 15,
  tableMaxCols: 12,
  tableMaxCellChars: 120,
});
console.log(overview.ndjson);

const sheets = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
console.log(sheets.ndjson);

const sheet = workbook.worksheets.getItemAt(0);
const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1.5, format: "png" });
await fs.writeFile(`${workDir}/input_preview.png`, new Uint8Array(await preview.arrayBuffer()));
