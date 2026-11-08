// Lint rules for the web app: ESLint's and typescript-eslint's recommended rules for every
// file, React's hooks rules for the TypeScript sources, and three adjustments for how this code
// is written. `npm run lint` passes --max-warnings=0, so a warning fails the check too.
import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig([
  // The build's output and generated clients aren't written by hand.
  globalIgnores(["**/dist/", "**/generated/"]),
  js.configs.recommended,
  tseslint.configs.recommended,
  {
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    rules: {
      // `ready && start()` and `ok ? a() : b()` are fine as statements.
      "@typescript-eslint/no-unused-expressions": [
        "error",
        { allowShortCircuit: true, allowTernary: true },
      ],
      // An empty catch is how this code says an error doesn't matter.
      "no-empty": ["error", { allowEmptyCatch: true }],
      // Prettier may break a line before "[" or "("; that never changes what the code means.
      "no-unexpected-multiline": "off",
    },
  },
  {
    files: ["**/*.{ts,tsx,mts,cts}"],
    plugins: { "react-hooks": reactHooks },
    rules: reactHooks.configs.recommended.rules,
  },
]);
