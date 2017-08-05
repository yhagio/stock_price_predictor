const markdownpdf = require("markdown-pdf");
const fs = require("fs");

fs.createReadStream("./report.md")
  .pipe(markdownpdf())
  .pipe(fs.createWriteStream("./report.pdf"));
