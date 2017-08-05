const markdownpdf = require("markdown-pdf");
const fs = require("fs");

fs.createReadStream("./proposal.md")
  .pipe(markdownpdf())
  .pipe(fs.createWriteStream("./proposal.pdf"));