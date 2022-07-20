import requests
import os
import argparse
import json

ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')

if __name__ == "__main__":

    parser = argparse.ArgumentParser('upload file to zenodo')
    parser.add_argument('--upload_file', dest='upload_file', type = str)
    args = parser.parse_args()

    headers = {"Content-Type": "application/json"}
    params = {'access_token': ACCESS_TOKEN}
    print(ACCESS_TOKEN)

    r = requests.post('https://zenodo.org/api/deposit/depositions',
                    params=params,
                    json={},
                    # Headers are not necessary here since "requests" automatically
                    # adds "Content-Type: application/json", because we're using
                    # the "json=" keyword argument
                    # headers=headers,
                    headers=headers)

    bucket_url = r.json()["links"]["bucket"]
    deposition_id = r.json()['id']
    filename = args.upload_file.rsplit('/', 1)[1]
    print('uploading data now...')
    with open(args.upload_file, "rb") as fp:
        r = requests.put(
            "%s/%s" % (bucket_url, filename),
            data=fp,
            params=params,
        )
    print('uploading finished')

    # data = {
    #      'metadata': {
    #          'title': 'MultiDCP data',
    #          'upload_type': 'poster',
    #          'description': 'MultiDCP data',
    #          'creators': [{'name': 'Qiao, Liu',
    #                        'affiliation': 'Hunter College, City University of New York'}]
    #      }
    #  }
    # r = requests.put('https://zenodo.org/api/deposit/depositions/%s' % deposition_id, 
    #     params={'access_token': ACCESS_TOKEN}, data=json.dumps(data),
    #     headers=headers)
    print(r.status_code)
