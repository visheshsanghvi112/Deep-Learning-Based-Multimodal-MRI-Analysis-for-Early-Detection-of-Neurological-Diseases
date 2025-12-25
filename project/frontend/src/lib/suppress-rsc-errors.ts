// Suppress harmless Next.js RSC 404 warnings in development console
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
    const originalError = console.error
    console.error = (...args) => {
        // Filter out harmless RSC-related 404s
        const errorString = args.join(' ')
        if (
            errorString.includes('?_rsc=') ||
            errorString.includes('Failed to load resource: the server responded with a status of 404') && errorString.includes('adni')
        ) {
            // Suppress these - they're harmless Next.js prefetch attempts
            return
        }
        originalError.apply(console, args)
    }
}

export { }
