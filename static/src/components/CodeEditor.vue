<template>
  <div class="editor">
    <pre class="gutter" ref="gutter"></pre>
    <textarea ref="ta" :value="modelValue" @input="onInput" @scroll="sync" @keydown.tab.prevent="insertTab" spellcheck="false" placeholder="在这里编辑 Python 代码…"></textarea>
  </div>
</template>

<script>
export default {
  name: 'CodeEditor',
  props: { modelValue: { type: String, default: '' } },
  emits: ['update:modelValue'],
  mounted() { this.$nextTick(this.sync); },
  updated() { this.sync(); },
  methods: {
    sync() {
      const ta = this.$refs.ta, g = this.$refs.gutter;
      if (!ta || !g) return;
      const lines = ta.value.split('\n').length;
      const nums = [];
      for (let i = 1; i <= lines; i++) nums.push(i);
      g.textContent = nums.join('\n');
      g.scrollTop = ta.scrollTop;
    },
    onInput(e) { this.$emit('update:modelValue', e.target.value); this.$nextTick(this.sync); },
    insertTab() {
      const ta = this.$refs.ta;
      const s = ta.selectionStart, e = ta.selectionEnd;
      const v = this.modelValue || '';
      this.$emit('update:modelValue', v.slice(0, s) + '  ' + v.slice(e));
      this.$nextTick(() => { ta.focus(); ta.selectionStart = ta.selectionEnd = s + 2; });
    }
  }
}
</script>
