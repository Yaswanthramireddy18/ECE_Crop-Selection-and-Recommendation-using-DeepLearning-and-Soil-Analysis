import os
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
    ConversationHandler,
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import pandas as pd

# Load crop data for N, P, K min/max values
crop_data = pd.read_csv(r"C:\New folder (3)\Dataset\xlfiles\Crop_recommendation 2.csv")

# Calculate min and max values from crop data
n_min, n_max = crop_data['N'].min(), crop_data['N'].max()
p_min, p_max = crop_data['P'].min(), crop_data['P'].max()
k_min, k_max = crop_data['K'].min(), crop_data['K'].max()

# Load the soil image model
image_model_path = r"C:\New folder (3)\custom_cnn_production_model.keras"
image_model = tf.keras.models.load_model(image_model_path, compile=False)

# Define soil types and default values
soil_types = ["Alluvial", "Black", "Clay", "Red"]
soil_defaults = {
    "Alluvial": {"npk": (90, 45, 40), "ph": 6.5},
    "Black": {"npk": (85, 35, 30), "ph": 7.2},
    "Clay": {"npk": (75, 25, 35), "ph": 6.8},
    "Red": {"npk": (80, 20, 25), "ph": 5.5}
}

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define conversation states
PHOTO, NPK, PH = range(3)

async def start(update: Update, context: CallbackContext) -> None:
    """Send welcome message and instructions"""
    user = update.effective_user
    await update.message.reply_text(
        f"👋 Hello {user.first_name}!\n\n"
        "I'm SoilAnalysisBot 🌱\n\n"
        "Send me a photo of soil to analyze, and I'll:\n"
        "1. Detect soil type\n"
        "2. Predict pH value\n"
        "3. Recommend best crops\n\n"
        "Just send a soil photo to get started!"
        "If you need help at any time, you can type `/help`."
    )

async def help_command(update: Update, context: CallbackContext) -> None:
    """Send help message explaining how to use the bot"""
    await update.message.reply_text(
        "ℹ️ **How to use SoilAnalysisBot** 🌱\n\n"
        "1. **Send a soil photo**: You can either capture a photo directly or choose one from your gallery.\n"
        "   - If you send a photo from your gallery, the bot will process it and detect the soil type.\n"
        "2. **Provide NPK values**: After detecting the soil type, you will be asked to provide NPK values (Nitrogen, Phosphorus, Potassium)."
        "   - You can type them as `N,P,K` or type 'skip' to use the default values for the detected soil type.\n"
        "3. **Provide or skip pH**: You can then provide a pH value (between 3.5 and 9.0) or skip it to let the bot predict it.\n"
        "4. **Receive recommendations**: The bot will then recommend the best crop based on the provided data.\n\n"
        "To start, just send a soil photo. If you need help at any time, you can type `/help`."
    )

async def handle_photo(update: Update, context: CallbackContext) -> int:
    """Handle received soil photo"""
    os.makedirs("temp_images", exist_ok=True)

    photo_file = await update.message.photo[-1].get_file()
    filename = os.path.join("temp_images", f"{update.message.message_id}.jpg")
    await photo_file.download_to_drive(filename)

    context.user_data["image_path"] = filename
    try:
        img = load_img(filename, target_size=(128, 128)) 
        img_array = img_to_array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

        soil_type_prediction = image_model.predict(img_array)
        soil_type_index = np.argmax(soil_type_prediction)
        detected_soil_type = soil_types[soil_type_index]

        await update.message.reply_text(
            f"📸 Photo received! Detected Soil Type: {detected_soil_type}\n\n"
            "Would you like to provide custom NPK values?\n"
            "(Send in format: N,P,K or type 'skip' to use defaults)"
        )

        context.user_data["soil_type"] = detected_soil_type

    except Exception as e:
        logger.error(f"Error processing photo: {e}")
        await update.message.reply_text(
            "⚠️ There was an error processing the image. Please try again."
        )

    return NPK

async def handle_npk(update: Update, context: CallbackContext) -> int:
    """Handle NPK input"""
    user_input = update.message.text.strip().lower()

    if user_input == "skip":
        context.user_data["npk"] = None
    else:
        try:
            n, p, k = map(float, user_input.split(","))
            context.user_data["npk"] = (n, p, k)
        except:
            await update.message.reply_text("⚠️ Invalid format! Please use N,P,K format")
            return NPK

    await update.message.reply_text(
        "🔬 Would you like to provide pH value?\n"
        "(Enter a number between 3.5-9.0 or 'skip' to use default)"
    )

    return PH

async def handle_ph(update: Update, context: CallbackContext) -> int:
    """Handle pH input and process analysis"""
    user_input = update.message.text.strip().lower()
    user_data = context.user_data
    soil_type = user_data["soil_type"]

    # Handle pH input
    ph = None
    if user_input != "skip":
        try:
            ph = float(user_input)
            if not 3.5 <= ph <= 9.0:
                raise ValueError
        except:
            await update.message.reply_text("⚠️ Invalid pH! Must be a number between 3.5 and 9.0")
            return PH

    # Get NPK values (use defaults if skipped)
    if user_data.get("npk") is None:
        default_npk = soil_defaults.get(soil_type, {}).get("npk", (75, 30, 30))
        n, p, k = default_npk
    else:
        n, p, k = user_data["npk"]

    # Get pH value (use default if skipped)
    if ph is None:
        ph = soil_defaults.get(soil_type, {}).get("ph", 6.5)

    try:
        recommended_crop = recommend_crop(n, p, k)

        result_text = (
            f"🌱 **Soil Analysis Results** 🌱\n\n"
            f"🔍 Detected Soil Type: {soil_type}\n"
            f"📊 pH Value: {ph:.1f}\n"
            f"🌾 Recommended Crop: {recommended_crop}\n\n"
            "Thank you for using SoilAnalysisBot!"
        )

        await update.message.reply_text(result_text)
        os.remove(user_data["image_path"])

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        await update.message.reply_text("⚠️ Error processing request. Please try again.")

    return ConversationHandler.END

def recommend_crop(N, P, K):
    """Recommend the best crop based on NPK values"""
    # Clip input values to dataset ranges
    N = np.clip(N, n_min, n_max)
    P = np.clip(P, p_min, p_max)
    K = np.clip(K, k_min, k_max)

    # Calculate the score based on the absolute difference between N, P, K values
    crop_data["score"] = (
        abs(crop_data["N"] - N) +
        abs(crop_data["P"] - P) +
        abs(crop_data["K"] - K)
    )

    # Find the crop with the smallest score (closest match)
    best_match = crop_data.loc[crop_data["score"].idxmin()]
    
    return best_match["label"]

def main() -> None:
    """Run the bot"""
    TOKEN = "7733442320:AAEb-Tfw-HdhUKThvj9UjTFt_wedDGnNnDo"
    app = Application.builder().token(TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("help", help_command))
    
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.PHOTO, handle_photo)],
        states={
            NPK: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_npk)],
            PH: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ph)],
        },
        fallbacks=[CommandHandler("cancel", lambda update, context: ConversationHandler.END)],
    )
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_handler)
    
    logger.info("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
