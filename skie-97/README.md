# ADS-B
Project to determine best representation for 
recent areas of poor gps navigational performance

functions.py scrapes the samples.adsbexchange.com site for sample data.
Recommended to just download one day's worth of data first (approx 85GB).

## Approach
- Conduct EDA on the data. Poor GPS performance can be a 
result of a multitude of factors, not all of which are due
to jamming or spoofing.
- The ADS-B system calculates nac_p based on a statistical algo?
Research should be done into how does this work.

## References:
1. https://mode-s.org/decode/content/ads-b/7-uncertainty.html#tb:sil-params
1. https://samples.adsbexchange.com/readsb-hist/
1. https://www.adsbexchange.com/version-2-api-wip/
1. https://easychair.org/publications/open/Mp3x
1. https://www.jetnet.com/blog/rising-gps-jamming-spoofing-threats-in-business-aviation.html