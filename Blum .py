from pyrogram import Client
from pyrogram.raw.functions.messages import RequestWebView
from urllib.parse import unquote
from utils.core import logger
from fake_useragent import UserAgent
from dotenv import load_dotenv
import aiohttp
import asyncio
import random
import os

# Загружаем переменные окружения
load_dotenv()

# Конфигурация
API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

ACC_DELAY = [5, 15]
WORKDIR = "sessions/"
POINTS = [100, 140]
SPEND_DIAMONDS = True
SLEEP_GAME_TIME = [30, 50]
MINI_SLEEP = [3, 7]
SLEEP_8HOURS = [60 * 60, 120 * 60]

class Blum:
    def __init__(self, thread: int, account: str, proxy_url: str = None):
        self.thread = thread
        self.name = account
        self.client = Client(name=account, api_id=API_ID, api_hash=API_HASH, workdir=WORKDIR)
        self.proxy_url = proxy_url

        self.auth_token = ""
        self.ref_token = ""
        headers = {'User-Agent': UserAgent(os='android').random}
        if self.proxy_url:
            self.session = aiohttp.ClientSession(headers=headers, connector=aiohttp.ProxyConnector(proxy=self.proxy_url, verify_ssl=False))
        else:
            self.session = aiohttp.ClientSession(headers=headers)

    async def main(self):
        await asyncio.sleep(random.randint(*ACC_DELAY))
        try:
            login = await self.login()
            if not login:
                await self.session.close()
                return 0
            logger.info(f"main | Thread {self.thread} | {self.name} | Start!")
        except Exception as err:
            logger.error(f"main | Thread {self.thread} | {self.name} | {err}")
            await self.session.close()
            return 0

        while True:
            try:
                valid = await self.is_token_valid()
                if not valid:
                    logger.warning(f"main | Thread {self.thread} | {self.name} | Token is invalid. Refreshing token...")
                    await self.refresh()
                await asyncio.sleep(random.randint(*MINI_SLEEP))

                await self.claim_diamond()
                await asyncio.sleep(random.randint(*MINI_SLEEP))

                try:
                    timestamp, start_time, end_time = await self.balance()
                except Exception as err:
                    logger.error(f"balance | Thread {self.thread} | {self.name} | {err}")
                    continue

                await self.get_referral_info()
                await asyncio.sleep(random.randint(*MINI_SLEEP))

                await self.do_tasks()
                await asyncio.sleep(random.randint(*MINI_SLEEP))

                if SPEND_DIAMONDS:
                    diamonds_balance = await self.get_diamonds_balance()
                    logger.info(f"main | Thread {self.thread} | {self.name} | Have {diamonds_balance} diamonds!")
                    for _ in range(diamonds_balance):
                        await self.game()
                        await asyncio.sleep(random.randint(*SLEEP_GAME_TIME))

                if start_time is None and end_time is None:
                    await self.start()
                    logger.info(f"main | Thread {self.thread} | {self.name} | Start farming!")
                elif start_time is not None and end_time is not None and timestamp >= end_time:
                    timestamp, balance = await self.claim()
                    logger.success(f"main | Thread {self.thread} | {self.name} | Claimed reward! Balance: {balance}")
                else:
                    add_sleep = random.randint(*SLEEP_8HOURS)
                    logger.info(f"main | Thread {self.thread} | {self.name} | Sleep {(end_time - timestamp + add_sleep)} seconds!")
                    await asyncio.sleep(end_time - timestamp + add_sleep)
                    await self.login()
                await asyncio.sleep(random.randint(*MINI_SLEEP))
            except Exception as err:
                logger.error(f"main | Thread {self.thread} | {self.name} | {err}")
                valid = await self.is_token_valid()
                if not valid:
                    logger.warning(f"main | Thread {self.thread} | {self.name} | Token is invalid. Refreshing token...")
                    await self.refresh()
                await asyncio.sleep(random.randint(*MINI_SLEEP))

    async def claim(self):
        try:
            async with self.session.post("https://game-domain.blum.codes/api/v1/farming/claim") as resp:
                resp_json = await resp.json()
                return int(resp_json.get("timestamp") / 1000), resp_json.get("availableBalance")
        except Exception as err:
            logger.error(f"claim | Thread {self.thread} | {self.name} | {err}")
            return None, None

    async def start(self):
        try:
            await self.session.post("https://game-domain.blum.codes/api/v1/farming/start")
        except Exception as err:
            logger.error(f"start | Thread {self.thread} | {self.name} | {err}")

    async def balance(self):
        try:
            async with self.session.get("https://game-domain.blum.codes/api/v1/user/balance") as resp:
                resp_json = await resp.json()
                timestamp = resp_json.get("timestamp")
                if resp_json.get("farming"):
                    start_time = resp_json.get("farming").get("startTime")
                    end_time = resp_json.get("farming").get("endTime")
                    return int(timestamp / 1000), int(start_time / 1000), int(end_time / 1000)
                return int(timestamp), None, None
        except Exception as err:
            logger.error(f"balance | Thread {self.thread} | {self.name} | {err}")
            return None, None, None

    async def login(self):
        try:
            tg_web_data = await self.get_tg_web_data()
            if not tg_web_data:
                return False
            json_data = {"query": tg_web_data}
            async with self.session.post("https://gateway.blum.codes/v1/auth/provider/PROVIDER_TELEGRAM_MINI_APP", json=json_data) as resp:
                resp_json = await resp.json()
                self.ref_token = resp_json.get("token").get("refresh")
                self.session.headers['Authorization'] = "Bearer " + resp_json.get("token").get("access")
                return True
        except Exception as err:
            logger.error(f"login | Thread {self.thread} | {self.name} | {err}")
            return False

    async def get_tg_web_data(self):
        await self.client.connect()
        try:
            web_view = await self.client.invoke(RequestWebView(
                peer=await self.client.resolve_peer('BlumCryptoBot'),
                bot=await self.client.resolve_peer('BlumCryptoBot'),
                platform='android',
                from_bot_menu=False,
                url='https://telegram.blum.codes/'
            ))
            auth_url = web_view.url
        except Exception as err:
            logger.error(f"get_tg_web_data | Thread {self.thread} | {self.name} | {err}")
            await self.client.disconnect()
            return False
        await self.client.disconnect()
        return unquote(string=unquote(string=auth_url.split('tgWebAppData=')[1].split('&tgWebAppVersion')[0]))

    async def get_referral_info(self):
        try:
            async with self.session.get("https://gateway.blum.codes/v1/friends/balance") as resp:
                resp_json = await resp.json()
                if resp_json['canClaim']:
                    claimed = await self.claim_referral()
                    logger.success(f"get_ref | Thread {self.thread} | {self.name} | Claimed referral reward! Claimed: {claimed}")
        except Exception as err:
            logger.error(f"get_referral_info | Thread {self.thread} | {self.name} | {err}")

    async def claim_referral(self):
        try:
            async with self.session.post("https://gateway.blum.codes/v1/friends/claim") as resp:
                return resp.json().get("claimed")
        except Exception as err:
            logger.error(f"claim_referral | Thread {self.thread} | {self.name} | {err}")

    async def do_tasks(self):
        try:
            async with self.session.get("https://game-domain.blum.codes/api/v1/tasks") as resp:
                resp_json = await resp.json()
                for task in resp_json:
                    if "subTasks" in task:
                        for subtask in task['subTasks']:
                            if subtask['status'] == "NOT_STARTED":
                                await self.session.post(f"https://game-domain.blum.codes/api/v1/tasks/{subtask['id']}/start")
                                logger.info(f"tasks | Thread {self.thread} | {self.name} | Summer Quest | TRY DO {subtask['title']} task!")
                                await asyncio.sleep(random.randint(*MINI_SLEEP))
                            elif subtask['status'] == "READY_FOR_CLAIM":
                                await self.session.post(f"https://game-domain.blum.codes/api/v1/tasks/{subtask['id']}/claim")
                                logger.success(f"tasks | Thread {self.thread} | {self.name} | Summer Quest | DONE {subtask['title']} task!")
                                await asyncio.sleep(random.randint(*MINI_SLEEP))
                    else:
                        if task['status'] == "NOT_STARTED":
                            await self.session.post(f"https://game-domain.blum.codes/api/v1/tasks/{task['id']}/start")
                            await asyncio.sleep(random.randint(*MINI_SLEEP))
                        elif task['status'] == "READY_FOR_CLAIM":
                            await self.session.post(f"https://game-domain.blum.codes/api/v1/tasks/{task['id']}/claim")
                            logger.success(f"tasks | Thread {self.thread} | {self.name} | Claimed TASK reward!")
                            await asyncio.sleep(random.randint(*MINI_SLEEP))
        except Exception as err:
            logger.error(f"do_tasks | Thread {self.thread} | {self.name} | {err}")

    async def is_token_valid(self):
        try:
            async with self.session.get("https://gateway.blum.codes/v1/user/me") as resp:
                return resp.status == 200
        except Exception as err:
            logger.error(f"is_token_valid | Thread {self.thread} | {self.name} | {err}")
            return False

    async def refresh(self):
        try:
            json_data = {"refresh": self.ref_token}
            async with self.session.post("https://gateway.blum.codes/v1/auth/refresh", json=json_data) as resp:
                if resp.status == 200:
                    resp_json = await resp.json()
                    self.session.headers['Authorization'] = "Bearer " + resp_json.get("access")
                    logger.info(f"refresh | Thread {self.thread} | {self.name} | Token refreshed successfully.")
                else:
                    logger.error(f"refresh | Thread {self.thread} | {self.name} | Failed to refresh token: {resp.status}")
        except Exception as err:
            logger.error(f"refresh | Thread {self.thread} | {self.name} | {err}")

    async def get_diamonds_balance(self):
        try:
            async with self.session.get("https://game-domain.blum.codes/api/v1/user/balance") as resp:
                resp_json = await resp.json()
                return resp_json['playPasses']
        except Exception as err:
            logger.error(f"get_diamonds_balance | Thread {self.thread} | {self.name} | {err}")
            return 0

    async def game(self):
        try:
            async with self.session.post('https://game-domain.blum.codes/api/v1/game/play') as resp:
                resp_json = await resp.json()
                if 'message' in resp_json:
                    logger.error(f"game | Thread {self.thread} | {self.name} | DROP GAME CAN'T START")
                    return
                text = resp_json['gameId']
                count = random.randint(*POINTS)
                await asyncio.sleep(30 if count < 160 else 30 + (count - 160) // 7 * 4)
                json_data = {'gameId': text, 'points': count}
                async with self.session.post('https://game-domain.blum.codes/api/v1/game/claim', json=json_data) as resp:
                    if await resp.text() == "OK":
                        logger.success(f"game | Thread {self.thread} | {self.name} | Claimed DROP GAME ! Claimed: {count}")
                    elif "Invalid jwt token" in await resp.text():
                        await self.refresh()
                    else:
                        logger.error(f"game | Thread {self.thread} | {self.name} | {await resp.text()}")
        except Exception as err:
            logger.error(f"game | Thread {self.thread} | {self.name} | {err}")

    async def claim_diamond(self):
        try:
            async with self.session.post("https://game-domain.blum.codes/api/v1/diamond/daily") as resp:
                await resp.json()
        except Exception as err:
            logger.error(f"claim_diamond | Thread {self.thread} | {self.name} | {err}")

    async def is_token_valid(self):
        try:
            async with self.session.get("https://gateway.blum.codes/v1/farming/validate_token") as resp:
                resp_json = await resp.json()
                return resp_json.get("valid")
        except Exception as err:
            logger.error(f"is_token_valid | Thread {self.thread} | {self.name} | {err}")
            return False

    async def refresh(self):
        try:
            json_data = {"refreshToken": self.ref_token}
            async with self.session.post("https://gateway.blum.codes/v1/auth/refresh", json=json_data) as resp:
                resp_json = await resp.json()
                self.session.headers['Authorization'] = "Bearer " + resp_json.get("access")
        except Exception as err:
            logger.error(f"refresh | Thread {self.thread} | {self.name} | {err}")

# Запуск основного процесса
async def main():
    account = "account_name"
    proxy_url = os.getenv('PROXY_URL')
    blum = Blum(thread=1, account=account, proxy_url=proxy_url)
    await blum.main()

if __name__ == '__main__':
    asyncio.run(main())
