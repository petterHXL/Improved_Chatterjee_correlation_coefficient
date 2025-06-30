#!/bin/bash
echo "🔄 Auto-pushing to GitHub..."
git add .
git commit -m "Auto-update: $(date)"
git push origin main
echo "✅ Successfully pushed to GitHub!"
