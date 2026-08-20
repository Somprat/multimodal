import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/macbooka/Desktop/maniskill_router_paraphrases_270.xlsx";
const workDir = "/Users/macbooka/Documents/VsCode/multimodal/.codex_work/task_router_300";

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const overview = await workbook.inspect({
  kind: "workbook,sheet,table,formula",
  maxChars: 16000,
  tableMaxRows: 14,
  tableMaxCols: 16,
  tableMaxCellChars: 120,
  options: { maxResults: 80 },
});
console.log(overview.ndjson);

for (let index = 0; index < 4; index += 1) {
  const sheet = workbook.worksheets.getItemAt(index);
  const preview = await workbook.render({
    sheetName: sheet.name,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const safeName = sheet.name.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(`${workDir}/input_${safeName}.png`, new Uint8Array(await preview.arrayBuffer()));
}
