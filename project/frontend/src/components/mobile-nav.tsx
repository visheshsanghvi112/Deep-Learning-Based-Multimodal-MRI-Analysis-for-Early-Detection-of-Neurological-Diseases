"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { motion, AnimatePresence } from "framer-motion"
import {
    Menu,
    X,
    Home,
    Database,
    Layers,
    Workflow,
    BarChart3,
    Brain,
    Map,
    FileText
} from "lucide-react"
import { createPortal } from "react-dom"

import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"

const links = [
    { href: "/", label: "Home", icon: Home },
    { href: "/dataset", label: "OASIS Dataset", icon: Database },
    { href: "/adni", label: "ADNI Validation", icon: Layers },
    { href: "/pipeline", label: "Pipeline", icon: Workflow },
    { href: "/results", label: "Results", icon: BarChart3 },
    { href: "/documentation", label: "Documentation", icon: FileText },
    { href: "/interpretability", label: "Interpretability", icon: Brain },
    { href: "/roadmap", label: "Roadmap", icon: Map },
]

export function MobileNav() {
    const [open, setOpen] = React.useState(false)
    const [mounted, setMounted] = React.useState(false)
    const pathname = usePathname()

    React.useEffect(() => {
        setMounted(true)
    }, [])

    React.useEffect(() => {
        if (open) {
            document.body.style.overflow = "hidden"
        } else {
            document.body.style.overflow = "unset"
        }
        return () => {
            document.body.style.overflow = "unset"
        }
    }, [open])

    return (
        <div className="md:hidden">
            <button
                onClick={() => setOpen(true)}
                className="flex items-center justify-center p-2 rounded-md hover:bg-muted transition-colors"
                aria-label="Open menu"
            >
                <Menu className="h-6 w-6" />
            </button>

            {mounted && createPortal(
                <AnimatePresence>
                    {open && (
                        <div className="fixed inset-0 z-[9999]">
                            {/* Backdrop */}
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                onClick={() => setOpen(false)}
                                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                            />

                            {/* Drawer */}
                            <motion.div
                                initial={{ x: "100%" }}
                                animate={{ x: 0 }}
                                exit={{ x: "100%" }}
                                transition={{ type: "spring", damping: 30, stiffness: 300 }}
                                className="absolute right-0 top-0 bottom-0 h-full w-[85%] max-w-sm border-l border-border bg-background shadow-2xl flex flex-col overflow-hidden"
                            >
                                {/* Decorative Gradient */}
                                <div className="absolute top-0 right-0 h-32 w-32 bg-primary/10 blur-3xl -z-10 rounded-full" />

                                <div className="flex items-center justify-between p-6 border-b bg-muted/20">
                                    <div className="flex items-center gap-2">
                                        <span className="text-lg font-semibold tracking-tight">NeuroScope</span>
                                    </div>
                                    <button
                                        onClick={() => setOpen(false)}
                                        className="p-2 rounded-full hover:bg-muted transition-colors"
                                        aria-label="Close menu"
                                    >
                                        <X className="h-5 w-5" />
                                    </button>
                                </div>

                                <div className="flex-1 overflow-y-auto py-6 px-4">
                                    <nav className="flex flex-col gap-2">
                                        {links.map((link, i) => {
                                            const active = pathname === link.href
                                            const Icon = link.icon

                                            return (
                                                <motion.div
                                                    key={link.href}
                                                    initial={{ opacity: 0, x: 20 }}
                                                    animate={{ opacity: 1, x: 0 }}
                                                    transition={{ delay: i * 0.05 }}
                                                >
                                                    <Link
                                                        href={link.href}
                                                        onClick={() => setOpen(false)}
                                                        className={cn(
                                                            "flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200",
                                                            active
                                                                ? "bg-primary/10 text-primary"
                                                                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                                                        )}
                                                    >
                                                        <Icon className={cn("h-4 w-4", active ? "text-primary" : "text-muted-foreground")} />
                                                        {link.label}
                                                    </Link>
                                                </motion.div>
                                            )
                                        })}
                                    </nav>
                                </div>

                                <div className="p-6 border-t bg-muted/20">
                                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                                        <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                                        <p>Systems operational</p>
                                    </div>
                                    <p className="mt-2 text-[10px] text-muted-foreground/60 uppercase tracking-wider">
                                        Research Portal · {new Date().getFullYear()}
                                    </p>
                                </div>
                            </motion.div>
                        </div>
                    )}
                </AnimatePresence>,
                document.body
            )}
        </div>
    )
}
