import { FormEvent, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Lock, ShieldCheck } from 'lucide-react'

import { Button } from './ui/button'
import { Input } from './ui/input'

export default function UILockScreen({
  visible,
  checking,
  timeoutMinutes,
  unlocking,
  error,
  onUnlock,
}: {
  visible: boolean
  checking: boolean
  timeoutMinutes: number
  unlocking: boolean
  error: string | null
  onUnlock: (password: string) => Promise<void> | void
}) {
  const [password, setPassword] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!visible || checking) return
    setPassword('')
    const timer = window.setTimeout(() => {
      inputRef.current?.focus()
    }, 120)
    return () => window.clearTimeout(timer)
  }, [visible, checking])

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const trimmed = password.trim()
    if (!trimmed) return
    await onUnlock(trimmed)
  }

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          className="fixed inset-0 z-[140] overflow-hidden"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25 }}
        >
          <div className="absolute inset-0 bg-slate-950/96" />
          <motion.div
            className="pointer-events-none absolute -top-24 left-[-8%] h-72 w-72 rounded-full bg-emerald-500/20 blur-3xl"
            initial={{ x: -40, y: -20, opacity: 0.4 }}
            animate={{ x: 30, y: 10, opacity: 0.8 }}
            transition={{ duration: 5.5, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' }}
          />
          <motion.div
            className="pointer-events-none absolute bottom-[-120px] right-[-10%] h-80 w-80 rounded-full bg-cyan-500/20 blur-3xl"
            initial={{ x: 40, y: 30, opacity: 0.35 }}
            animate={{ x: -20, y: -10, opacity: 0.75 }}
            transition={{ duration: 6.2, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' }}
          />

          <div className="relative flex h-full items-center justify-center p-4">
            <motion.div
              initial={{ y: 24, scale: 0.97, opacity: 0 }}
              animate={{ y: 0, scale: 1, opacity: 1 }}
              exit={{ y: 16, scale: 0.98, opacity: 0 }}
              transition={{ duration: 0.28, ease: 'easeOut' }}
              className="w-full max-w-md rounded-2xl border border-emerald-500/20 bg-slate-900/85 p-6 shadow-[0_28px_120px_rgba(16,185,129,0.18)] backdrop-blur-md"
            >
              <div className="mb-5 flex items-center gap-3">
                <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-2.5">
                  <Lock className="h-5 w-5 text-emerald-300" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-100">Homerun Locked</h2>
                  <p className="text-xs text-slate-400">
                    Inactivity timeout: {timeoutMinutes} minute{timeoutMinutes === 1 ? '' : 's'}
                  </p>
                </div>
              </div>

              {checking ? (
                <div className="flex items-center gap-2 text-sm text-slate-300">
                  <ShieldCheck className="h-4 w-4 text-emerald-300" />
                  Checking lock session...
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-3">
                  <Input
                    ref={inputRef}
                    type="password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="Enter password"
                    autoComplete="current-password"
                    className="h-10 border-slate-700 bg-slate-950/60 text-slate-100 placeholder:text-slate-500"
                  />
                  {error ? <p className="text-xs text-red-300">{error}</p> : null}
                  <Button
                    type="submit"
                    disabled={unlocking || password.trim().length === 0}
                    className="w-full h-10 bg-emerald-600 text-white hover:bg-emerald-500"
                  >
                    {unlocking ? 'Unlocking...' : 'Unlock'}
                  </Button>
                </form>
              )}
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
