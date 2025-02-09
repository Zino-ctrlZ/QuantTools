"""
This is used to test the proxy server works to fetch thetadata 
"""

import http.client
import json

conn = http.client.HTTPConnection("18.232.166.224", 5500)
payload = json.dumps({
  "method": "GET",
  "url": "http://127.0.0.1:25510/v2/hist/option/eod?exp=20231103&right=C&strike=170000&start_date=20231103&end_date=20231103&root=AAPL"
})
headers = {
  'Accept': 'application/json',
  'Content-Type': 'application/json'
}
conn.request("POST", "/thetadata", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))