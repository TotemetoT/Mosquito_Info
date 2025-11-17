async function loadCSV() {
    const response = await fetch("Mosquito.csv");
    const data = await response.text();

    // Split rows while respecting quoted commas
    const rows = data.split("\n").map(line => {
        return line.match(/(".*?"|[^",]+)(?=\s*,|\s*$)/g);
    });

    const table = document.getElementById("mosquito-table");

    // Build header row
    let header = document.createElement("tr");
    rows[0].forEach(col => {
        let th = document.createElement("th");
        th.textContent = col.replace(/"/g, "").trim();
        header.appendChild(th);
    });
    table.appendChild(header);

    const headers = rows[0]; // needed for column count

    // Build data rows
    rows.slice(1).forEach((row, index) => {
        if (!row) return;

        // Create row
        const tr = document.createElement("tr");

        // ⭐ Apply alternating row colors
        tr.classList.add(index % 2 === 0 ? "even-row" : "odd-row");

        // ⭐ Fix striping breaking by filling missing columns
        while (row.length < headers.length) {
            row.push("");
        }

        row.forEach((col, colIndex) => {
            let td = document.createElement("td");
            col = (col || "").replace(/"/g, "").trim();

            // Column 1 = disease list
            if (colIndex === 1) {
                let list = document.createElement("ul");

                if (col.length === 0) {
                    let li = document.createElement("li");
                    li.textContent = "None";
                    list.appendChild(li);
                } else {
                    col.split(",").forEach(item => {
                        let li = document.createElement("li");
                        li.textContent = item.trim();
                        list.appendChild(li);
                    });
                }

                td.appendChild(list);
            } else {
                td.textContent = col;
            }

            tr.appendChild(td);
        });

        table.appendChild(tr);
    });
}

loadCSV();
