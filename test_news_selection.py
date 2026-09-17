import unittest
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime
from news_selection import select_news


class NewsSelectionTests(unittest.TestCase):
    now = datetime(2026, 9, 17, tzinfo=timezone.utc)

    def entry(self, title, days=1, source='Reuters'):
        return {'title': title, 'published': format_datetime(self.now - timedelta(days=days)),
                'source': {'title': source}, 'link': title, 'summary': ''}

    def select(self, entries):
        return select_news(entries, '삼성전자', ('Reuters',), limit=4, now=self.now)

    def test_prefers_earnings_over_product_news(self):
        result = self.select([self.entry('Samsung Galaxy phone colors revealed'),
                              self.entry('Samsung Electronics chip profit and revenue surge')])
        self.assertEqual(len(result), 1)
        self.assertIn('profit', result[0][0]['title'])

    def test_keeps_negative_production_news(self):
        self.assertEqual(len(self.select([self.entry('Samsung workers strike disrupts production')])), 1)

    def test_excludes_wrong_company_source_and_dates(self):
        entries = [self.entry('Apple chip profit rises'), self.entry('Samsung SDI revenue rises'),
                   self.entry('Samsung chip profit rises', 15),
                   self.entry('Samsung chip profit rises', -1),
                   self.entry('Samsung chip profit rises', source='Unknown')]
        entries.append(dict(self.entry('Samsung chip profit rises'), published='unknown'))
        self.assertEqual(self.select(entries), [])

    def test_recency_and_duplicate_titles(self):
        old = self.entry('Samsung chip earnings increase', 10)
        new = self.entry('Samsung chip profit increases', 1)
        result = self.select([old, new, new.copy()])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], new)


if __name__ == '__main__':
    unittest.main()
