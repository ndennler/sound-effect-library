import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

for page_num in tqdm(range(1,13)):
    URL = f"https://notificationsounds.com/wake-up-tones/pages/{page_num}"
    r = requests.get(URL)
    
    soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib

    quotes=[]  # a list to store quotes
    
    table = soup.find_all('article', attrs = {'class':'flex flex-col justify-start bg-white rounded overflow-hidden shadow-lg mb-8 xl:mb-0'}) 

    for sound in table:
        name = sound.find("a", attrs = {'class':'block text-2xl text-link mb-6 leading-6'})
        filename = sound.find('source', attrs={'type': 'audio/mpeg'})
        if '_' in filename['src']:
            fname = filename['src'].split('_')[-1]
        else:
            fname = '_'.join(filename['src'].split('-')[3:])

        headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0"}
        try:
            with open(f"raw_audio_files/{fname}", 'wb') as f:
                f.write(requests.get(f"https://notificationsounds.com{filename['src']}", headers=headers).content)
        except Exception as e:
            print(e)

