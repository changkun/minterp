# The website of the notes: built by bookkit, the engine in a checkout beside
# this repository (see book.config.ts), and published at changkun.de/minterp.
# A published build reads the last commit, never the working tree.

HOST ?= changkunde
DEST ?= /www/changkun.de/minterp/
SNAPSHOT ?= .cache/snapshot
BOOKKIT = node node_modules/bookkit/bin/bookkit.ts

node_modules: package.json
	npm install

# The working tree, served at http://localhost:4400 and rebuilt on change.
dev: node_modules
	npm run dev

# A copy of the last commit, with the installed engine linked in.
snapshot: node_modules
	rm -rf $(SNAPSHOT) && mkdir -p $(SNAPSHOT)
	git archive HEAD | tar -x -C $(SNAPSHOT)
	ln -s $(CURDIR)/node_modules $(SNAPSHOT)/node_modules

site: snapshot
	cd $(SNAPSHOT) && $(BOOKKIT) build --out $(CURDIR)/_site

# Every page in a browser at two widths, the layout at every width from a
# phone to a wide screen, and the fixed header under an elastic scroll.
check: site
	$(BOOKKIT) shots --all
	$(BOOKKIT) sweep index.html 08-limits.html src/scripts/patch.py.html --base /minterp/
	$(BOOKKIT) bounce 08-limits.html --base /minterp/

deploy: check
	rsync -az --delete _site/ $(HOST):$(DEST)
	@echo "published to https://changkun.de/minterp/"

.PHONY: dev snapshot site check deploy
