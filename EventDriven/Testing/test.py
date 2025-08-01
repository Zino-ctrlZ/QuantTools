"""
This is used to test the proxy server works to fetch thetadata 
"""

import http.client
import json
import os

# print(os.environ['PROXY_URL'])
# {'end_date': 20250619, 'root': 'AAPL', 'use_csv': 'true', 'exp': 20241220, 'right': 'C', 'start_date': 20170101, 'strike': 220000, 'url': 'http://127.0.0.1:25510/v2/hist/option/eod?end_date=20250619&root=AAPL&use_csv=true&exp=20241220&right=C&start_date=20170101&strike=220000'}
conn = http.client.HTTPConnection("54.144.4.219", 5500)
payload = json.dumps({
  "method": "GET",
  "url": 'http://127.0.0.1:25510/v2/hist/option/eod?end_date=20250619&root=AAPL&use_csv=true&exp=20241220&right=C&start_date=20240101&strike=220000'
})
url_old = "http://127.0.0.1:25510/v2/hist/option/eod?exp=20231103&right=C&strike=170000&start_date=20231103&end_date=20231103&root=AAPL"
headers = {
  'Accept': 'application/json',
  'Content-Type': 'application/json'
}
conn.request("POST", "/thetadata", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))

print("\n\n\n")
retrieve_quote_payload = json.dumps({
  "method": "GET", 
  "url": "http://127.0.0.1:25510/v2/hist/option/quote?end_date=20230706&root=MSFT&use_csv=true&exp=20240621&ivl=1800000&right=C&start_date=20230706&strike=355000&start_time=34200000&rth=False&end_time=57600000"
})
conn.request("POST", "/thetadata", retrieve_quote_payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
conn.close()