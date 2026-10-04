// Make the MAE and PSNR columns of the leaderboard tables sortable.
//
// Every sortable header gets a button: the first click sorts the column from
// best to worst (ascending MAE, descending PSNR), the following ones reverse
// the order. The Rank column is renumbered after each sort. The leaderboards
// are generated already sorted by decreasing PSNR.
(function () {
  "use strict";

  // header pattern -> order putting the best value first
  const SORTABLE = [
    [/^MAE\b/, "ascending"],
    [/^PSNR\b/, "descending"],
  ];

  function value(row, col) {
    const v = parseFloat(row.cells[col].textContent);
    return Number.isNaN(v) ? -Infinity : v;
  }

  function setup(table) {
    const headers = Array.from(table.tHead.rows[0].cells);
    const labels = headers.map((th) => th.textContent.trim());
    if (labels[0] !== "Rank") return; // leaderboards only
    const body = table.tBodies[0];
    const buttons = [];

    headers.forEach((th, col) => {
      const match = SORTABLE.find(([re]) => re.test(labels[col]));
      if (!match) return;
      const best = match[1];
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sort-button";
      button.innerHTML = `${labels[col]}<span class="sort-arrow" aria-hidden="true"></span>`;
      th.replaceChildren(button);
      th.setAttribute("aria-sort", "none");
      buttons.push(th);

      button.addEventListener("click", () => {
        const current = th.getAttribute("aria-sort");
        const order =
          current === "none" ? best : current === "ascending" ? "descending" : "ascending";
        const sign = order === "ascending" ? 1 : -1;
        const rows = Array.from(body.rows);
        rows.sort((a, b) => sign * (value(a, col) - value(b, col)));
        rows.forEach((row, i) => {
          row.cells[0].textContent = String(i + 1);
          body.appendChild(row);
        });
        buttons.forEach((other) => other.setAttribute("aria-sort", "none"));
        th.setAttribute("aria-sort", order);
      });

      if (/^PSNR\b/.test(labels[col])) th.setAttribute("aria-sort", "descending");
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("table").forEach((table) => {
      if (table.tHead && table.tBodies.length) setup(table);
    });
  });
})();
