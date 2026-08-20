import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/macbooka/Downloads/maniskill_instructions_to_label.xlsx";
const workDir = "/Users/macbooka/Documents/VsCode/multimodal/.codex_work/temporal_instructions";

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const summary = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 10000,
  tableMaxRows: 20,
  tableMaxCols: 16,
  tableMaxCellChars: 120,
});
console.log(summary.ndjson);

const firstSheet = workbook.worksheets.getItemAt(0);
const usedRange = firstSheet.getUsedRange();
const usedAddress = usedRange?.address ?? "A1:Z40";
console.log(`first_sheet=${firstSheet.name}`);
console.log(`used_range=${usedAddress}`);

const styles = await workbook.inspect({
  kind: "computedStyle",
  sheetId: firstSheet.name,
  range: usedAddress,
  maxChars: 8000,
});
console.log(styles.ndjson);

const preview = await workbook.render({
  sheetName: firstSheet.name,
  autoCrop: "all",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(`${workDir}/input_preview.png`, new Uint8Array(await preview.arrayBuffer()));
