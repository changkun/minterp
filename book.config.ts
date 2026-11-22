// The website of the notes, built by bookkit: the engine in a checkout beside
// this repository (../bookkit), whose README documents every key below.
//
//   make site      build _site/ from the last commit
//   make dev       build the working tree and serve it

import { defineBook } from "bookkit/config.ts";

export default defineBook({
  bookkit: "542a388e2214db5d35c5e9fba8883818cb483484",
  title: "minterp",
  subtitle: "Interpretability by hand, on a model you can train",
  author: "Changkun Ou",
  repo: "https://github.com/changkun/minterp",
  url: "https://changkun.de/minterp/",
  // The notes in reading order, as they are in docs/. The landing page is the README.
  content: "docs",
  pages: ["01-read.md", "02-probe.md", "03-patch.md", "04-circuit.md", "05-features.md", "06-feature-circuit.md", "07-steer.md", "08-limits.md", "09-stack.md"],
  home: "README.md",
  // A link from a note to a file of this repository becomes a page showing the
  // file. A link to a directory, such as runs/patch/, leads to it on GitHub.
  sources: { branch: "main", directories: true },
  // The figures are placed into the page as markup, so the theme switch reaches
  // them, and each opens large with pan and zoom.
  images: { inline: true, zoom: true },
  header: { logo: "site/logo.svg" },
  // The accent is the blue of the figures.
  theme: { accent: "#0b6fb8", "accent-ink": "#09588f", "accent-soft": "#e8f2fa", dark: { accent: "#58b0f0", "accent-ink": "#8cc8f5", "accent-soft": "#16293a" } },
  footer: {
    license: false,
    stats: false,
    author: { name: "Changkun Ou", url: "https://changkun.de" },
    org: { name: "Latere AI", url: "https://latere.ai", mark: "site/latere-mark.svg" },
  },
  series: {
    name: "AI safety",
    items: [
      { label: "1", title: "Scalable Oversight", url: "https://changkun.de/scalable-oversight/" },
      { label: "2", title: "AI Alignment", url: "https://changkun.de/alignbook/" },
      { label: "3", title: "minterp", current: true },
      { label: "+", title: "minllm", url: "https://minllm.latere.site/" },
    ],
  },
});
