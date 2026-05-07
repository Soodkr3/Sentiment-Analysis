FROM node:18-alpine

WORKDIR /app

COPY package.json ./
# --legacy-peer-deps handles react-bootstrap peer dep range
RUN npm install --legacy-peer-deps

COPY . .

EXPOSE 3000

# Note: react-scripts start requires source files under src/ (Stage 2 restructuring).
# This image currently installs deps and is ready for a volume-mount dev workflow.
CMD ["npm", "start"]
