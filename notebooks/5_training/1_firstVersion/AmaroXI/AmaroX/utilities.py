from telegram import Bot
import asyncio

async def send_sms_to_me(sms: str) -> None:
    '''
    Description
        This function makes use of the Amaro-TelegramBot to send a message when the code is finished. 

            await send_sms_to_me('Hi')

    Args
        sms (str): Refers to the message to be send 
   '''
    BOT_TOKEN =''
    CHAT_ID = 0
    bot = Bot(token = BOT_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text = sms)
