/**
 * MarketCard — OpenUI custom component for rendering rich market cards
 * in AI chat responses. Styled to match the chat's tool result widgets.
 */

import { defineComponent } from '@openuidev/react-lang'
import { z } from 'zod'
import { useTranslation } from 'react-i18next'
import {
  TrendingUp,
  TrendingDown,
  Clock,
  BarChart3,
  Droplets,
  ExternalLink,
} from 'lucide-react'

export const MarketCard = defineComponent({
  name: 'MarketCard',
  props: z.object({
    question: z.string().describe('Market question / title'),
    slug: z.string().describe('Polymarket event slug for link construction'),
    yes_price: z.number().describe('YES price in cents (e.g. 11 for 11c)'),
    no_price: z.number().describe('NO price in cents (e.g. 89 for 89c)'),
    volume: z.string().describe('Formatted volume (e.g. "$277K")'),
    liquidity: z.string().describe('Formatted liquidity (e.g. "$30K")'),
    end_date: z.string().describe('Expiration / resolution date (e.g. "Dec 31, 2026")'),
    commentary: z.string().optional().describe('Agent analysis or edge description'),
  }),
  description:
    'Rich market opportunity card with prices, metadata, and a link to Polymarket. ' +
    'Use when presenting individual market opportunities or recommendations.',
  component: ({ props }) => {
    const { t } = useTranslation()
    const {
      question,
      slug,
      yes_price,
      no_price,
      volume,
      liquidity,
      end_date,
      commentary,
    } = props

    const polymarketUrl = `https://polymarket.com/event/${slug}`
    const yesIsLeading = yes_price >= no_price

    return (
      <div className="my-1.5 rounded-lg border border-purple-500/20 bg-purple-500/5 overflow-hidden">
        {/* Header */}
        <div className="px-3 py-2 border-b border-purple-500/10">
          <p className="text-xs font-medium text-foreground/90 leading-snug">
            {question}
          </p>
        </div>

        {/* Price pills */}
        <div className="px-3 py-2 flex items-center gap-3 flex-wrap">
          <span
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-bold ${
              yesIsLeading
                ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                : 'bg-emerald-500/10 text-emerald-400/70 border border-emerald-500/15'
            }`}
          >
            <TrendingUp className="w-3 h-3" />
            {t('aiMarketCard.yesPrice', { price: yes_price })}
          </span>
          <span
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-bold ${
              !yesIsLeading
                ? 'bg-red-500/15 text-red-400 border border-red-500/30'
                : 'bg-red-500/10 text-red-400/70 border border-red-500/15'
            }`}
          >
            <TrendingDown className="w-3 h-3" />
            {t('aiMarketCard.noPrice', { price: no_price })}
          </span>
        </div>

        {/* Metadata row */}
        <div className="px-3 py-1.5 flex items-center gap-4 flex-wrap border-t border-purple-500/10 bg-muted/10">
          <span className="inline-flex items-center gap-1 text-[10px] text-muted-foreground/70">
            <BarChart3 className="w-3 h-3" />
            {t('aiMarketCard.volume', { value: volume })}
          </span>
          <span className="inline-flex items-center gap-1 text-[10px] text-muted-foreground/70">
            <Droplets className="w-3 h-3" />
            {t('aiMarketCard.liquidity', { value: liquidity })}
          </span>
          <span className="inline-flex items-center gap-1 text-[10px] text-muted-foreground/70">
            <Clock className="w-3 h-3" />
            {t('aiMarketCard.ends', { date: end_date })}
          </span>
        </div>

        {/* Commentary */}
        {commentary && (
          <div className="px-3 py-2 border-t border-purple-500/10">
            <p className="text-[11px] text-muted-foreground/80 leading-relaxed">
              {commentary}
            </p>
          </div>
        )}

        {/* Action row */}
        <div className="px-3 py-1.5 border-t border-purple-500/10 flex items-center justify-end">
          <a
            href={polymarketUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium text-purple-400 hover:text-purple-300 hover:bg-purple-500/10 transition-colors"
          >
            <ExternalLink className="w-3 h-3" />
            {t('aiMarketCard.viewOnPolymarket')}
          </a>
        </div>
      </div>
    )
  },
})
