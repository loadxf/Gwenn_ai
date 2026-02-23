---
{
  "name": "summarize_url",
  "description": "Fetches a web page or article and returns a concise summary of its content. Use when someone shares a URL and asks what it contains, wants a summary or overview, asks 'what does this say?', 'can you read this?', or 'what are the key points from this link?'",
  "category": "information",
  "version": "1.1",
  "risk_level": "medium",
  "tags": ["user_command", "summarize", "url", "article", "web page", "read", "link"],
  "parameters": {
    "url": {
      "type": "string",
      "description": "The full URL of the page to fetch and summarise (must start with http:// or https://)",
      "required": true
    },
    "focus": {
      "type": "string",
      "description": "Specific aspect to focus on in the summary — defaults to a general overview of key points",
      "default": "key points"
    }
  }
}
---

Fetches the content at **{url}** and summarises it, focusing on **{focus}**.

## Steps

1. Call `fetch_url` with `url="{url}"` and `max_chars=8000`.
   - Record the `Content-Type` from the response header.

2. Identify the content type and clean accordingly:
   - **HTML page**: Locate the main body — look for content in `<article>`, `<main>`, or `<p>` tags. Ignore navigation bars, sidebars, footers, cookie banners, and repeated boilerplate.
   - **JSON API response**: Call `format_json` with `action="format"`, then describe the structure and key data.
   - **Plain text / Markdown**: Read directly with no processing.
   - **RSS/XML feed**: Extract `<title>` and `<description>` from `<item>` elements.

3. Write the summary with these sections:

   **What it is** — One sentence: what type of content and its subject.

   **Key points** — 3–7 bullet points covering the {focus}. Each bullet should be a standalone insight, not a fragment.

   **Notable details** — Any statistics, quotes, dates, names, or conclusions worth highlighting.

   **Source note** — The page title (if found), the domain, and whether the content was truncated.

## Output format

Aim for 150–250 words. Use bullet points for key points; prose for the opening and source note.

## Quality rules

- Never invent or infer content that wasn't in the fetched text
- If the page is paywalled (content < 200 words after cleaning), say so explicitly and explain what is visible
- If the URL returns an error, report the HTTP status code and suggest the user check the link
- If content was truncated (fetch_url returns a truncation notice), flag this and offer to fetch with a higher `max_chars`
- If the page is in another language, translate key points into the same language as the conversation
