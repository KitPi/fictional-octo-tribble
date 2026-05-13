import asyncio
import time

import aiohttp
import numpy as np
import rasterio

rasterPaths = [
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_360519_S1Hand.tif",
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_379434_S1Hand.tif",
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_294583_S1Hand.tif",
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_195474_S1Hand.tif",
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_314919_S1Hand.tif",
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_312675_S1Hand.tif",
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_242570_S1Hand.tif",
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_290290_S1Hand.tif",
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_233925_S1Hand.tif",
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_103757_S1Hand.tif",
    "/home/kit/Documents/QGIS/RemoteSensing/files/S1/Bolivia_129334_S1Hand.tif",
]


async def send_single_request(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.json()


async def send_concurrent_requests():
    images = []
    for im in rasterPaths:
        with rasterio.open(im) as src
            images.append({"vv": src.read(1).tolist(), "vh": src.read(2).tolist()})

    print(f"Sending {len(rasterPaths)} concurrent requests...")
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_single_request(
                session, url="http://localhost:8000/", data={"vv": image["vv"], "vh": image["vh"]}
            )
            for image in images
        ]
        responses = await asyncio.gather(*tasks)

    return responses


# Run the concurrent requests.
start_time = time.time()
responses = asyncio.run(send_concurrent_requests())
elapsed = time.time() - start_time

print(f"Processed {len(responses)} requests in {elapsed:.2f} seconds")
print(f"Throughput: {len(responses) / elapsed:.2f} requests/second")
