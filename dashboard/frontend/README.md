# SupplySim Dashboard Frontend

## Run

```bash
cd dashboard/frontend
npm install
npm run dev
```

Frontend expects backend at `http://localhost:8000` by default.
Override with:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

## Tests

```bash
npm run test
npm run test:e2e
```
