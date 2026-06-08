from flask import Flask, request, jsonify
from gradio_client import Client
import os

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")

try:
    client = Client("yisol/IDM-VTON", hf_token=HF_TOKEN)
except Exception as e:
    print(f"Error: {e}")

@app.route('/api/virtual-tryon', methods=['POST'])
def virtual_tryon():
    try:
        data = request.json
        person_image_url = data.get('person_image')
        cloth_image_url = data.get('cloth_image')
        clothing_type = data.get('cloth_type', 'Upper body')

        if not person_image_url or not cloth_image_url:
            return jsonify({"status": "error", "message": "Missing images"}), 400

        result = client.predict(
            dict={"background": person_image_url, "layers": [], "composite": None},
            garm_img=cloth_image_url,
            category=clothing_type,
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        return jsonify({
            "status": "success",
            "result_image": result[0] if isinstance(result, tuple) else result
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
