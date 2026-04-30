
CVE Dataset version used: [2026-04-29_all_CVEs_at_midnight.zip.zip](https://github.com/CVEProject/cvelistV5/releases/download/cve_2026-04-29_1800Z/2026-04-29_all_CVEs_at_midnight.zip.zip)

1. download CVE dataset V5 from cve.org: wget <URI> and unzip it into a cveV5 directory in this directory.
2. run the fine-tuning script "01" to generate the "domain adapted model".
3. run "test 17" script with the given parameters

It is possible to re-run the hyperparameters sweep using script "02".