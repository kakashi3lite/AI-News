# RSE News Dashboard - Markdown Style Guide

## Overview

This style guide ensures consistent formatting and structure across all Markdown files in the RSE News Dashboard project. Following these guidelines improves readability, maintainability, and professional presentation.

## Table of Contents

- [File Structure](#file-structure)
- [Headers](#headers)
- [Text Formatting](#text-formatting)
- [Lists](#lists)
- [Links and References](#links-and-references)
- [Code Blocks](#code-blocks)
- [Tables](#tables)
- [Images](#images)
- [Quotes and Citations](#quotes-and-citations)
- [Line Length and Spacing](#line-length-and-spacing)
- [Special Elements](#special-elements)
- [News Article Format](#news-article-format)
- [Documentation Format](#documentation-format)

## File Structure

### File Naming
- Use lowercase with hyphens: `file-name.md`
- Be descriptive: `api-documentation.md` not `docs.md`
- Use consistent prefixes for related files: `guide-`, `api-`, `tutorial-`

### Front Matter (if applicable)
```yaml
---
title: "Document Title"
date: 2024-01-15
author: "Author Name"
tags: ["tag1", "tag2"]
description: "Brief description of the document"
---
```

## Headers

### Hierarchy
- Use only one H1 (`#`) per document
- Follow logical hierarchy: H1 → H2 → H3 → H4
- Skip header levels sparingly and only when necessary

### Formatting
```markdown
# Main Title (H1)

## Section Title (H2)

### Subsection Title (H3)

#### Sub-subsection Title (H4)
```

### Guidelines
- Use sentence case: "Getting started with the API"
- Avoid punctuation at the end of headers
- Keep headers concise and descriptive
- Use consistent terminology throughout

## Text Formatting

### Emphasis
- **Bold** for strong emphasis: `**important text**`
- *Italic* for mild emphasis: `*emphasized text*`
- ***Bold italic*** for very strong emphasis: `***critical text***`

### Inline Code
- Use backticks for code elements: `variable`, `function()`, `file.js`
- Use for file names, commands, and technical terms

### Strikethrough
- Use for deprecated content: `~~deprecated feature~~`

## Lists

### Unordered Lists
```markdown
- First item
- Second item
  - Nested item
  - Another nested item
- Third item
```

### Ordered Lists
```markdown
1. First step
2. Second step
   1. Sub-step
   2. Another sub-step
3. Third step
```

### Guidelines
- Use consistent bullet characters (`-` for unordered)
- Use proper indentation (2 spaces) for nested items
- End list items with periods only if they are complete sentences
- Keep list items parallel in structure

## Links and References

### Internal Links
```markdown
[Link text](./relative/path/to/file.md)
[Section link](#section-header)
```

### External Links
```markdown
[Link text](https://example.com)
[Link with title](https://example.com "Tooltip text")
```

### Reference Links
```markdown
[Link text][reference-id]

[reference-id]: https://example.com "Optional title"
```

### Guidelines
- Use descriptive link text (avoid "click here")
- Verify all links are working
- Use HTTPS when available
- Group reference links at the bottom of the document

## Code Blocks

### Fenced Code Blocks
```markdown
```javascript
const example = 'Hello, World!';
console.log(example);
```
```

### Language Specification
- Always specify the language for syntax highlighting
- Common languages: `javascript`, `python`, `bash`, `json`, `yaml`, `markdown`

### Inline vs Block
- Use inline code for short snippets: `const x = 5`
- Use code blocks for multi-line code or complete examples

## Tables

### Basic Table
```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |
```

### Alignment
```markdown
| Left | Center | Right |
|:-----|:------:|------:|
| L1   |   C1   |    R1 |
| L2   |   C2   |    R2 |
```

### Guidelines
- Keep tables simple and readable
- Use alignment for better visual organization
- Consider breaking complex tables into multiple simpler ones

## Images

### Basic Image
```markdown
![Alt text](./images/image.png)
```

### Image with Title
```markdown
![Alt text](./images/image.png "Image title")
```

### Guidelines
- Always include meaningful alt text
- Use relative paths for local images
- Optimize image sizes for web
- Store images in organized directories

## Quotes and Citations

### Blockquotes
```markdown
> This is a blockquote.
> It can span multiple lines.
>
> And include multiple paragraphs.
```

### Nested Quotes
```markdown
> Main quote
>> Nested quote
> Back to main quote
```

### Citations
```markdown
According to the research[^1], this approach is effective.

[^1]: Smith, J. (2024). "Research Title". Journal Name, 15(3), 123-145.
```

## Line Length and Spacing

### Line Length
- Aim for 80-100 characters per line
- Break long lines at natural points (after punctuation)
- Use soft wrapping in editors

### Spacing
- One blank line between paragraphs
- Two blank lines before major sections (H2)
- One blank line before/after lists, code blocks, and tables
- No trailing whitespace

## Special Elements

### Horizontal Rules
```markdown
---
```

### Task Lists
```markdown
- [x] Completed task
- [ ] Incomplete task
- [ ] Another task
```

### Alerts (GitHub-style)
```markdown
> [!NOTE]
> This is a note.

> [!WARNING]
> This is a warning.

> [!IMPORTANT]
> This is important information.
```

## News Article Format

### Template
```markdown
# Article Title

**Source:** [Source Name](source-url) | **Published:** YYYY-MM-DD | **Topics:** topic1, topic2

## Summary

Brief 2-3 sentence summary of the article.

## Key Points

- Important point 1
- Important point 2
- Important point 3

## Content

Main article content with proper formatting...

## Impact

Analysis of the potential impact or significance.

## Related Articles

- [Related Article 1](link)
- [Related Article 2](link)

---

**Tags:** #tag1 #tag2 #tag3
**Quality Score:** X.X/5.0
**Processing Time:** XXXms
```

## Documentation Format

### API Documentation
```markdown
# API Endpoint Name

## Overview

Brief description of what this endpoint does.

## Request

### Method
`GET` | `POST` | `PUT` | `DELETE`

### URL
```
/api/v1/endpoint
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| param1    | string | Yes | Description |
| param2    | number | No | Description |

### Example Request
```json
{
  "param1": "value1",
  "param2": 123
}
```

## Response

### Success Response
```json
{
  "status": "success",
  "data": {}
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Error description"
}
```
```

## Validation Rules

The RSE Quality Auditor will check for:

### Required Elements
- [ ] Single H1 header
- [ ] Proper header hierarchy
- [ ] No trailing whitespace
- [ ] Consistent list formatting
- [ ] Valid links
- [ ] Alt text for images

### Style Consistency
- [ ] Sentence case headers
- [ ] Consistent emphasis formatting
- [ ] Proper code block language specification
- [ ] Table alignment
- [ ] Consistent spacing

### Content Quality
- [ ] Descriptive link text
- [ ] Meaningful alt text
- [ ] Complete sentences in lists (when appropriate)
- [ ] Proper citation format
- [ ] No broken links

## Tools and Automation

### Linting
- Use `markdownlint` for automated checking
- Configure rules in `.markdownlint.json`
- Integrate with pre-commit hooks

### Formatting
- Use `prettier` for consistent formatting
- Configure in `.prettierrc`
- Set up editor integration

### Validation
- Link checking with automated tools
- Spell checking integration
- Grammar checking for content quality

## Examples

### Good Example
```markdown
# Getting Started with the RSE News API

## Overview

The RSE News API provides programmatic access to our curated collection of research software engineering news articles.

## Authentication

All API requests require authentication using an API key:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.rse-news.com/v1/articles
```

### Rate Limits

- **Free tier:** 100 requests per hour
- **Pro tier:** 1,000 requests per hour
- **Enterprise:** Custom limits

## Quick Start

1. [Register for an API key](https://rse-news.com/api/register)
2. Make your first request
3. Explore the available endpoints

For more information, see our [complete API documentation](./api-reference.md).
```

### Bad Example
```markdown
# getting started

this is the api documentation

## authentication
You need an api key. Get one here: https://example.com

```
curl https://api.example.com
```

### rate limits
Don't make too many requests!!!

- free: 100/hour
- pro:1000/hour
- enterprise:unlimited

Click [here](./docs.md) for more info.
```

## Conclusion

Following this style guide ensures that all Markdown content in the RSE News Dashboard project maintains high quality, consistency, and professionalism. The RSE Quality Auditor will automatically check for compliance with these guidelines during the CI/CD process.

For questions or suggestions about this style guide, please [open an issue](https://github.com/your-repo/issues) or contact the development team.