/**
 * Theme extraction configuration constants.
 */

const THEME_CATEGORIES = {
  technology: ['AI', 'machine learning', 'blockchain', 'cryptocurrency', 'tech', 'software', 'hardware'],
  politics: ['election', 'government', 'policy', 'congress', 'senate', 'president', 'political'],
  business: ['market', 'stock', 'economy', 'finance', 'company', 'earnings', 'investment'],
  health: ['health', 'medical', 'disease', 'vaccine', 'hospital', 'doctor', 'treatment'],
  environment: ['climate', 'environment', 'green', 'renewable', 'carbon', 'pollution', 'sustainability'],
  sports: ['sports', 'game', 'team', 'player', 'championship', 'league', 'tournament'],
  entertainment: ['movie', 'music', 'celebrity', 'entertainment', 'film', 'show', 'actor']
};

const SENTIMENT_KEYWORDS = {
  positive: ['breakthrough', 'success', 'achievement', 'growth', 'improvement', 'victory', 'progress'],
  negative: ['crisis', 'failure', 'decline', 'problem', 'issue', 'concern', 'controversy'],
  neutral: ['report', 'analysis', 'study', 'research', 'data', 'information', 'update']
};

const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
  'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
  'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
  'said', 'says', 'new', 'also', 'more', 'most', 'some', 'many', 'much', 'very', 'just',
  'now', 'then', 'here', 'there', 'where', 'when', 'how', 'why', 'what', 'who', 'which'
]);

module.exports = { THEME_CATEGORIES, SENTIMENT_KEYWORDS, STOP_WORDS };
