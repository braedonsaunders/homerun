import { AlertCircle } from 'lucide-react'
import { cn } from '../lib/utils'

interface OpportunityEmptyStateProps {
  title: string
  description?: string
  className?: string
}

export default function OpportunityEmptyState({
  title,
  description,
  className,
}: OpportunityEmptyStateProps) {
  return (
    <div className={cn('text-center py-12 bg-card rounded-lg border border-border', className)}>
      <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
      <p className="text-muted-foreground">{title}</p>
      {description ? <p className="text-sm text-muted-foreground/50 mt-1">{description}</p> : null}
    </div>
  )
}
