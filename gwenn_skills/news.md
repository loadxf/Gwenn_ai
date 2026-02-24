---
{
  "name": "get_news",
  "description": "Fetches and summarises recent news headlines on any topic, person, company, or event. Use when someone asks about current events, what's happening in the world, latest news, breaking news, recent developments, or anything that happened today or this week.",
  "category": "information",
  "version": "1.1",
  "risk_level": "medium",
  "tags": ["user_command", "news", "headlines", "current events", "breaking news"],
  "parameters": {
    "topic": {
      "type": "string",
      "description": "Topic, keyword, person, company, or event to search news for (e.g. 'AI', 'climate', 'Apple', 'Ukraine')",
      "required": true
    },
    "count": {
      "type": "integer",
      "description": "Number of headlines to return (1–10)",
      "default": 5
    }
  }
}
---

Fetches and presents the latest {count} news headlines about **{topic}**.

## Steps

1. URL-encode the topic before embedding it in the URL: replace spaces with `+` and percent-encode special characters (`&` → `%26`, `#` → `%23`, `=` → `%3D`, etc.). For example, `"AI & climate"` becomes `"AI+%26+climate"`.

2. Call `fetch_url` with URL: `https://news.google.com/rss/search?q=<url-encoded-topic>&hl=en-GB&gl=GB&ceid=GB:en`
   - Returns an RSS/XML feed of recent Google News results
   - Look for `<item>` elements — each contains `<title>`, `<pubDate>`, and `<link>`

3. Parse up to **{count}** items from the feed:
   - Extract the `<title>` content (strip surrounding CDATA markers if present)
   - Extract the `<pubDate>` for recency context
   - Extract the source name if present (often appended to the title after ` - `)

4. If the Google News feed is unavailable or returns no results, fall back to:
   `https://feeds.bbci.co.uk/news/rss.xml` (BBC World News — no topic filtering)

5. Present the headlines in a clean numbered list. After the list, add 1–2 sentences summarising the dominant theme or story across the results.

## Output format

```
Here are the latest {count} headlines on **{topic}**:

1. **[Headline]** — [Source], [relative date e.g. "2 hours ago" or "yesterday"]
2. **[Headline]** — [Source], [date]
...

The main story right now is [brief theme summary].
```

## Quality rules

- Present at most {count} items — never pad with repetitive or duplicate headlines
- Note if all results are from the same source (possible bias)
- Flag if the most recent result is more than 24 hours old — news may be stale
- Never fabricate headlines; only report what the feed actually returns
