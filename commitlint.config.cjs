const CLAUDE_TRAILER_PATTERNS = [
  /co-authored-by:[^\n]*claude/i,
  /claude-session:/i,
  /noreply@anthropic\.com/i,
];

module.exports = {
  extends: ["@commitlint/config-conventional"],
  plugins: [
    {
      rules: {
        "no-claude-trailers": (parsed) => {
          const text = `${parsed.raw || ""}\n${parsed.footer || ""}`;
          const bad = CLAUDE_TRAILER_PATTERNS.some((re) => re.test(text));
          return [!bad, "commit must not name Claude as author or co-author"];
        },
      },
    },
  ],
  rules: {
    "no-claude-trailers": [2, "always"],
    "type-enum": [
      2,
      "always",
      [
        "feat",
        "fix",
        "docs",
        "infra",
        "refactor",
        "test",
        "chore",
        "perf",
        "build",
        "ci",
        "revert",
      ],
    ],
  },
};
