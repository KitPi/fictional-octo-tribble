import asyncio
import os
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
]


async def send_single_request(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.json()


async def send_concurrent_requests():
    images = []
    for im in rasterPaths:
        with rasterio.open(im) as src:
            vv = np.nan_to_num(src.read(1), nan=0.0)
            vh = np.nan_to_num(src.read(2), nan=0.0)
            images.append({"vv": vv.tolist(), "vh": vh.tolist()})

    print(f"Sending {len(rasterPaths)} concurrent requests...")
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_single_request(
                session,
                url="http://localhost:8000/",
                data={"vv": image["vv"], "vh": image["vh"]},
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


# from rasterio.transform import from_origin

os.makedirs("responses", exist_ok=True)

for count, (path, response) in enumerate(zip(rasterPaths, responses)):
    with rasterio.open(path) as src:
        meta = src.profile
        meta.update(count=1, dtype="float32")

    with rasterio.open(f"responses/response_{count}_new.tif", "w", **meta) as dst:
        dst.write(np.array(response, dtype=np.float32), 1)
