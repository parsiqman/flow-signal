# Weather forward collection

## Nothing snapshotted

No temperature bands parsed out of 0 markets returned by /public-search. Either none are running, or the outcome wording has drifted from what parse_band accepts -- the sample below is the thing to check.

## What is actually open right now

The parser wants a city from `weather.CITIES`, the word temperature/temp/degrees, and a band it can read. If these questions carry the band as a separate OUTCOME rather than in the question text, the parser needs to read outcomes too.


