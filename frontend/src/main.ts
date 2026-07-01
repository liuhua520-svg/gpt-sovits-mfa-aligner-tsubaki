import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import { getElementPlusLocale, i18n } from './i18n'
import { router } from './router'

const app = createApp(App)

app.use(i18n)
app.use(router)
app.use(ElementPlus, {
  locale: getElementPlusLocale(i18n.global.locale.value),
})
app.mount('#app')
