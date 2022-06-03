import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CallbackContext, CommandHandler

TELEGRAM_BOT_TOKEN = "5456013033:AAGgT8tllsN06qBAvwzB03dbrEVmopNn5vc"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: CallbackContext.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm biodiversipy's bot!")

if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    application.run_polling()
