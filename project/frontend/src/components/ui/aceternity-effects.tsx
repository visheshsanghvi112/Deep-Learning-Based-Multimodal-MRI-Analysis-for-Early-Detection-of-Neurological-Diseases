"use client"

import { useRef, useState, useEffect } from "react"
import { motion, useMotionTemplate, useMotionValue, useSpring, useInView, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"

// =============================================================================
// SPOTLIGHT CARD - Dramatic hover spotlight effect
// =============================================================================
export function SpotlightCard({
    children,
    className,
    spotlightColor = "rgba(0, 212, 255, 0.15)",
}: {
    children: React.ReactNode
    className?: string
    spotlightColor?: string
}) {
    const divRef = useRef<HTMLDivElement>(null)
    const [isFocused, setIsFocused] = useState(false)
    const [position, setPosition] = useState({ x: 0, y: 0 })
    const [opacity, setOpacity] = useState(0)

    const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!divRef.current) return
        const rect = divRef.current.getBoundingClientRect()
        setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top })
    }

    const handleMouseEnter = () => {
        setIsFocused(true)
        setOpacity(1)
    }

    const handleMouseLeave = () => {
        setIsFocused(false)
        setOpacity(0)
    }

    return (
        <div
            ref={divRef}
            className={cn("relative overflow-hidden rounded-xl border border-border/50 bg-card", className)}
            onMouseMove={handleMouseMove}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
        >
            <div
                className="pointer-events-none absolute -inset-px opacity-0 transition-opacity duration-300"
                style={{
                    opacity,
                    background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, ${spotlightColor}, transparent 40%)`,
                }}
            />
            {children}
        </div>
    )
}

// =============================================================================
// GLOWING BORDER CARD - Animated gradient border
// =============================================================================
export function GlowingBorderCard({
    children,
    className,
    containerClassName,
    glowColor = "from-cyan-500 via-purple-500 to-pink-500",
}: {
    children: React.ReactNode
    className?: string
    containerClassName?: string
    glowColor?: string
}) {
    return (
        <div className={cn("group relative p-[1px] rounded-xl overflow-hidden", containerClassName)}>
            {/* Animated gradient border */}
            <div
                className={cn(
                    "absolute inset-0 rounded-xl bg-gradient-to-r opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-sm",
                    glowColor
                )}
            />
            <div
                className={cn(
                    "absolute inset-0 rounded-xl bg-gradient-to-r opacity-30 group-hover:opacity-60 transition-opacity duration-500",
                    glowColor
                )}
                style={{
                    animation: "spin 3s linear infinite",
                }}
            />
            {/* Inner content */}
            <div className={cn("relative rounded-xl bg-card", className)}>{children}</div>
        </div>
    )
}

// =============================================================================
// TEXT GRADIENT ANIMATION - Shimmering text effect
// =============================================================================
export function TextGradient({
    children,
    className,
    colors = "from-cyan-400 via-purple-500 to-pink-500",
    animate = true,
}: {
    children: React.ReactNode
    className?: string
    colors?: string
    animate?: boolean
}) {
    return (
        <span
            className={cn(
                "bg-clip-text text-transparent bg-gradient-to-r bg-[length:200%_auto]",
                colors,
                animate && "animate-text-shimmer",
                className
            )}
        >
            {children}
        </span>
    )
}

// =============================================================================
// ANIMATED COUNTER - Number counting animation
// =============================================================================
export function AnimatedCounter({
    value,
    suffix = "",
    prefix = "",
    duration = 2,
    className,
    decimals = 0,
}: {
    value: number
    suffix?: string
    prefix?: string
    duration?: number
    className?: string
    decimals?: number
}) {
    const ref = useRef<HTMLSpanElement>(null)
    const isInView = useInView(ref, { once: true, margin: "-100px" })
    const [displayValue, setDisplayValue] = useState(0)

    useEffect(() => {
        if (!isInView) return

        let startTime: number
        let animationFrame: number

        const animate = (timestamp: number) => {
            if (!startTime) startTime = timestamp
            const progress = Math.min((timestamp - startTime) / (duration * 1000), 1)

            // Easing function (ease out cubic)
            const easeOut = 1 - Math.pow(1 - progress, 3)
            setDisplayValue(easeOut * value)

            if (progress < 1) {
                animationFrame = requestAnimationFrame(animate)
            }
        }

        animationFrame = requestAnimationFrame(animate)
        return () => cancelAnimationFrame(animationFrame)
    }, [isInView, value, duration])

    return (
        <span ref={ref} className={className}>
            {prefix}
            {displayValue.toFixed(decimals)}
            {suffix}
        </span>
    )
}

// =============================================================================
// 3D HOVER CARD - Tilt effect on hover (disabled on touch devices)
// =============================================================================
export function Card3D({
    children,
    className,
    containerClassName,
}: {
    children: React.ReactNode
    className?: string
    containerClassName?: string
}) {
    const ref = useRef<HTMLDivElement>(null)
    const [isTouchDevice, setIsTouchDevice] = useState(false)

    // Detect touch device on mount
    useEffect(() => {
        setIsTouchDevice('ontouchstart' in window || navigator.maxTouchPoints > 0)
    }, [])

    const x = useMotionValue(0)
    const y = useMotionValue(0)

    const xSpring = useSpring(x, { stiffness: 200, damping: 20 })
    const ySpring = useSpring(y, { stiffness: 200, damping: 20 })

    const transform = useMotionTemplate`perspective(1000px) rotateX(${xSpring}deg) rotateY(${ySpring}deg)`

    const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
        if (isTouchDevice || !ref.current) return
        const rect = ref.current.getBoundingClientRect()
        const width = rect.width
        const height = rect.height
        const mouseX = e.clientX - rect.left
        const mouseY = e.clientY - rect.top
        const rotateX = ((mouseY - height / 2) / height) * -8
        const rotateY = ((mouseX - width / 2) / width) * 8
        x.set(rotateX)
        y.set(rotateY)
    }

    const handleMouseLeave = () => {
        x.set(0)
        y.set(0)
    }

    // On touch devices, just render children without transform
    if (isTouchDevice) {
        return <div className={cn("relative", containerClassName, className)}>{children}</div>
    }

    return (
        <div className={cn("relative", containerClassName)} style={{ perspective: 1000 }}>
            <motion.div
                ref={ref}
                className={cn("relative", className)}
                style={{ transform, transformStyle: "preserve-3d" }}
                onMouseMove={handleMouseMove}
                onMouseLeave={handleMouseLeave}
            >
                {children}
            </motion.div>
        </div>
    )
}

// =============================================================================
// FLOATING PARTICLES - Background decoration
// =============================================================================
export function FloatingParticles({
    count = 50,
    className,
}: {
    count?: number
    className?: string
}) {
    const particles = Array.from({ length: count }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.random() * 4 + 1,
        duration: Math.random() * 20 + 10,
        delay: Math.random() * 5,
    }))

    return (
        <div className={cn("absolute inset-0 overflow-hidden pointer-events-none", className)}>
            {particles.map((particle) => (
                <motion.div
                    key={particle.id}
                    className="absolute rounded-full bg-primary/20"
                    style={{
                        left: `${particle.x}%`,
                        top: `${particle.y}%`,
                        width: particle.size,
                        height: particle.size,
                    }}
                    animate={{
                        y: [-20, 20, -20],
                        opacity: [0.3, 0.8, 0.3],
                    }}
                    transition={{
                        duration: particle.duration,
                        repeat: Infinity,
                        delay: particle.delay,
                        ease: "easeInOut",
                    }}
                />
            ))}
        </div>
    )
}

// =============================================================================
// GRID BACKGROUND - Animated dot grid
// =============================================================================
export function GridBackground({
    className,
    children,
}: {
    className?: string
    children?: React.ReactNode
}) {
    return (
        <div className={cn("relative", className)}>
            <div
                className="absolute inset-0 dark:opacity-40 opacity-[0.03]"
                style={{
                    backgroundImage: `radial-gradient(circle at 1px 1px, currentColor 1px, transparent 0)`,
                    backgroundSize: "40px 40px",
                }}
            />
            {children}
        </div>
    )
}

// =============================================================================
// REVEAL ANIMATION - Scroll-triggered reveal
// =============================================================================
export function RevealOnScroll({
    children,
    className,
    delay = 0,
    direction = "up",
}: {
    children: React.ReactNode
    className?: string
    delay?: number
    direction?: "up" | "down" | "left" | "right"
}) {
    const ref = useRef<HTMLDivElement>(null)
    const isInView = useInView(ref, { once: true, margin: "-50px" })

    const directions = {
        up: { y: 40, x: 0 },
        down: { y: -40, x: 0 },
        left: { y: 0, x: 40 },
        right: { y: 0, x: -40 },
    }

    return (
        <motion.div
            ref={ref}
            className={className}
            initial={{ opacity: 0, ...directions[direction] }}
            animate={isInView ? { opacity: 1, x: 0, y: 0 } : { opacity: 0, ...directions[direction] }}
            transition={{ duration: 0.6, delay, ease: "easeOut" }}
        >
            {children}
        </motion.div>
    )
}

// =============================================================================
// MAGNETIC BUTTON - Button that follows mouse
// =============================================================================
export function MagneticButton({
    children,
    className,
    strength = 0.3,
}: {
    children: React.ReactNode
    className?: string
    strength?: number
}) {
    const ref = useRef<HTMLDivElement>(null)
    const x = useMotionValue(0)
    const y = useMotionValue(0)

    const xSpring = useSpring(x, { stiffness: 300, damping: 20 })
    const ySpring = useSpring(y, { stiffness: 300, damping: 20 })

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!ref.current) return
        const rect = ref.current.getBoundingClientRect()
        const centerX = rect.left + rect.width / 2
        const centerY = rect.top + rect.height / 2
        x.set((e.clientX - centerX) * strength)
        y.set((e.clientY - centerY) * strength)
    }

    const handleMouseLeave = () => {
        x.set(0)
        y.set(0)
    }

    return (
        <motion.div
            ref={ref}
            className={cn("inline-block", className)}
            style={{ x: xSpring, y: ySpring }}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
        >
            {children}
        </motion.div>
    )
}

// =============================================================================
// BEAM LINES - Animated connecting lines
// =============================================================================
export function BeamLines({ className }: { className?: string }) {
    return (
        <div className={cn("absolute inset-0 overflow-hidden pointer-events-none", className)}>
            <svg className="absolute inset-0 w-full h-full opacity-20 dark:opacity-30">
                <defs>
                    <linearGradient id="beam-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="transparent" />
                        <stop offset="50%" stopColor="currentColor" />
                        <stop offset="100%" stopColor="transparent" />
                    </linearGradient>
                </defs>
                {[...Array(5)].map((_, i) => (
                    <motion.line
                        key={i}
                        x1="0%"
                        y1={`${20 + i * 15}%`}
                        x2="100%"
                        y2={`${20 + i * 15}%`}
                        stroke="url(#beam-gradient)"
                        strokeWidth="1"
                        initial={{ pathLength: 0, opacity: 0 }}
                        animate={{
                            pathLength: [0, 1, 0],
                            opacity: [0, 0.5, 0],
                        }}
                        transition={{
                            duration: 3,
                            repeat: Infinity,
                            delay: i * 0.5,
                            ease: "easeInOut",
                        }}
                    />
                ))}
            </svg>
        </div>
    )
}

// =============================================================================
// TYPING EFFECT - Text typing animation
// =============================================================================
export function TypingEffect({
    text,
    className,
    speed = 50,
    delay = 0,
}: {
    text: string
    className?: string
    speed?: number
    delay?: number
}) {
    const [displayText, setDisplayText] = useState("")
    const [isTyping, setIsTyping] = useState(false)
    const ref = useRef<HTMLSpanElement>(null)
    const isInView = useInView(ref, { once: true })

    useEffect(() => {
        if (!isInView) return

        const timeout = setTimeout(() => {
            setIsTyping(true)
            let i = 0
            const interval = setInterval(() => {
                if (i < text.length) {
                    setDisplayText(text.slice(0, i + 1))
                    i++
                } else {
                    clearInterval(interval)
                    setIsTyping(false)
                }
            }, speed)
            return () => clearInterval(interval)
        }, delay)

        return () => clearTimeout(timeout)
    }, [isInView, text, speed, delay])

    return (
        <span ref={ref} className={className}>
            {displayText}
            {isTyping && <span className="animate-pulse">|</span>}
        </span>
    )
}
